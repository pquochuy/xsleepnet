import tensorflow as tf
from nn_basic_layers import *
from filterbank_shape import FilterbankShape
from ops import *

class XSleepNet(object):

    def __init__(self, config):
        # Placeholders for input, output and dropout
        self.config = config
        self.input_x1 = tf.placeholder(tf.float32,shape=[None, self.config.epoch_seq_len, self.config.deep_ntime, self.config.nchannel],name='input_x1')
        self.input_x2 = tf.placeholder(tf.float32, [None, self.config.epoch_seq_len, self.config.seq_frame_seq_len, self.config.seq_ndim, self.config.nchannel], name="input_x2")
        self.input_y = tf.placeholder(tf.float32, [None, self.config.epoch_seq_len, self.config.nclass], name="input_y")
        self.dropout_rnn = tf.placeholder(tf.float32, name="dropout_rnn")
        self.dropout_cnn = tf.placeholder(tf.float32, name="dropout_cnn")
        self.istraining = tf.placeholder(tf.bool, name='istraining') # idicate training for batch normmalization

        self.seq_frame_seq_len = tf.placeholder(tf.int32, [None]) # for the dynamic RNN
        self.epoch_seq_len = tf.placeholder(tf.int32, [None]) # for the dynamic RNN

        self.w1 = tf.placeholder(tf.float32, name="w1")
        self.w2 = tf.placeholder(tf.float32, name="w2")
        self.w3 = tf.placeholder(tf.float32, name="w3")

        self.construct_all_cnn_net()
        self.construct_seqsleepnet()
        self.construct_joint_branch()

        # summed loss
        self.output_loss = self.w1*self.deep_loss + self.w2*self.seq_loss + self.w3*self.joint_loss

        # add on regularization excepts for the filter bank layers
        with tf.name_scope("l2_loss"):
            vars   = tf.trainable_variables()
            except_vars_eeg = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='seq_filterbank-layer-eeg')
            except_vars_eog = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='seq_filterbank-layer-eog')
            except_vars_emg = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='seq_filterbank-layer-emg')
            l2_loss = tf.add_n([ tf.nn.l2_loss(v) for v in vars
                    if v not in except_vars_eeg and v not in except_vars_eog and v not in except_vars_emg])
            self.loss = self.output_loss + self.config.l2_reg_lambda*l2_loss

        self.accuracy = []
        self.score = []
        self.predictions = []
        # Accuracy
        with tf.name_scope("accuracy"):
            for i in range(self.config.epoch_seq_len):
                score_i = self.w1*tf.nn.softmax(self.deep_scores[i], name='deep_softmax') + \
                                  self.w2*tf.nn.softmax(self.seq_scores[i], name='seq_softmax') + \
                                  self.w3*tf.nn.softmax(self.joint_scores[i], name='joint_softmax')
                pred_i = tf.argmax(score_i, 1, name="pred-%s" % i)
                self.score.append(score_i)
                self.predictions.append(pred_i)
                correct_prediction_i = tf.equal(self.predictions[i], tf.argmax(tf.squeeze(self.input_y[:,i,:]), 1))
                accuracy_i = tf.reduce_mean(tf.cast(correct_prediction_i, "float"), name="accuracy-%s" % i)
                self.accuracy.append(accuracy_i)

    #############################################################
    # Construct fcnnrnn here
    #############################################################
    def construct_all_cnn_net(self):
        # num fmaps for all CNNs
        self.g_enc_depths = [16, 16, 32, 32, 64, 64, 128, 128, 256]
        deep_X = tf.reshape(self.input_x1, [-1, self.config.deep_ntime, self.config.nchannel])

        with tf.variable_scope("deep_all_cnn_layer") as scope:
            deep_cnn_feat = self.all_convolution_block(deep_X,"all_conv_block")
            deep_num_cnn_feat = 6*self.g_enc_depths[-1]
            deep_cnn_feat = tf.reshape(deep_cnn_feat, [-1, deep_num_cnn_feat])
            print("deep_cnn_feat")
            print(deep_cnn_feat.get_shape())
            deep_rnn_input = tf.reshape(deep_cnn_feat, [-1, self.config.epoch_seq_len, deep_num_cnn_feat])

        # bidirectional seq-level recurrent layer
        with tf.variable_scope("deep_epoch_rnn_layer") as scope:
            deep_fw_cell, deep_bw_cell = bidirectional_recurrent_layer(self.config.deep_nhidden,
                                                                           self.config.deep_nlayer,
                                                                           input_keep_prob=self.dropout_rnn,
                                                                           output_keep_prob=self.dropout_rnn)
            self.deep_rnn_out, self.deep_rnn_state = bidirectional_recurrent_layer_output(deep_fw_cell,
                                                                                deep_bw_cell,
                                                                                deep_rnn_input,
                                                                                self.epoch_seq_len,
                                                                                scope=scope)
            print(self.deep_rnn_out.get_shape())

        self.deep_scores = []
        self.deep_predictions = []
        with tf.variable_scope("deep_output_layer"):
            for i in range(self.config.epoch_seq_len):
                deep_score_i = fc(tf.squeeze(self.deep_rnn_out[:,i,:]),
                                self.config.deep_nhidden * 2,
                                self.config.nclass,
                                name="deep_output",
                                relu=False)
                deep_pred_i = tf.argmax(deep_score_i, 1, name="deep_pred-%s" % i)
                self.deep_scores.append(deep_score_i)
                self.deep_predictions.append(deep_pred_i)

        self.deep_loss = 0
        with tf.name_scope("deep-loss"):
            for i in range(self.config.epoch_seq_len):
                deep_loss_i = tf.nn.softmax_cross_entropy_with_logits(labels=tf.squeeze(self.input_y[:,i,:]), logits=self.deep_scores[i])
                deep_loss_i = tf.reduce_sum(deep_loss_i, axis=[0])
                self.deep_loss += deep_loss_i
        self.deep_loss = self.deep_loss/self.config.epoch_seq_len # average over sequence length

        self.deep_accuracy = []
        # Accuracy
        with tf.name_scope("deep-accuracy"):
            for i in range(self.config.epoch_seq_len):
                deep_correct_prediction_i = tf.equal(self.deep_predictions[i], tf.argmax(tf.squeeze(self.input_y[:,i,:]), 1))
                deep_accuracy_i = tf.reduce_mean(tf.cast(deep_correct_prediction_i, "float"), name="deep-accuracy-%s" % i)
                self.deep_accuracy.append(deep_accuracy_i)
        #############################################################
        # End deepsleepnet here
        #############################################################

    #############################################################
    # Construct seqsleepnet here
    #############################################################
    def construct_seqsleepnet(self):
        filtershape = FilterbankShape()
        #triangular filterbank
        self.Wbl = tf.constant(filtershape.lin_tri_filter_shape(nfilt=self.config.seq_nfilter,
                                                                nfft=self.config.seq_nfft,
                                                                samplerate=self.config.seq_samplerate,
                                                                lowfreq=self.config.seq_lowfreq,
                                                                highfreq=self.config.seq_highfreq),
                               dtype=tf.float32,
                               name="W-filter-shape-eeg")

        with tf.variable_scope("seq_filterbank-layer-eeg"):
            # Temporarily crush the feature_mat's dimensions
            Xeeg = tf.reshape(tf.squeeze(self.input_x2[:,:,:,:,0]), [-1, self.config.seq_ndim])
            # first filter bank layer
            self.Weeg = tf.Variable(tf.random_normal([self.config.seq_ndim, self.config.seq_nfilter],dtype=tf.float32))
            # non-negative constraints
            self.Weeg = tf.sigmoid(self.Weeg)
            # mask matrix should be replaced by shape-specific filter bank, e.g. triangular,rectangle.
            self.Wfb = tf.multiply(self.Weeg,self.Wbl)
            HWeeg = tf.matmul(Xeeg, self.Wfb) # filtering
            HWeeg = tf.reshape(HWeeg, [-1, self.config.epoch_seq_len, self.config.seq_frame_seq_len, self.config.seq_nfilter])

        if(self.config.nchannel > 1):
            with tf.variable_scope("seq_filterbank-layer-eog"):
                # Temporarily crush the feature_mat's dimensions
                Xeog = tf.reshape(tf.squeeze(self.input_x2[:,:,:,:,1]), [-1, self.config.seq_ndim])
                # first filter bank layer
                self.Weog = tf.Variable(tf.random_normal([self.config.seq_ndim, self.config.seq_nfilter],dtype=tf.float32))
                # non-negative constraints
                self.Weog = tf.sigmoid(self.Weog)
                # mask matrix should be replaced by shape-specific filter bank, e.g. triangular,rectangle.
                self.Wfb = tf.multiply(self.Weog,self.Wbl)
                HWeog = tf.matmul(Xeog, self.Wfb) # filtering
                HWeog = tf.reshape(HWeog, [-1, self.config.epoch_seq_len, self.config.seq_frame_seq_len, self.config.seq_nfilter])

        if(self.config.nchannel > 2):
            with tf.variable_scope("seq_filterbank-layer-emg"):
                # Temporarily crush the feature_mat's dimensions
                Xemg = tf.reshape(tf.squeeze(self.input_x2[:,:,:,:,2]), [-1, self.config.seq_ndim])
                # first filter bank layer
                self.Wemg = tf.Variable(tf.random_normal([self.config.seq_ndim, self.config.seq_nfilter],dtype=tf.float32))
                # non-negative constraints
                self.Wemg = tf.sigmoid(self.Wemg)
                # mask matrix should be replaced by shape-specific filter bank, e.g. triangular,rectangle.
                self.Wfb = tf.multiply(self.Wemg,self.Wbl)
                HWemg = tf.matmul(Xemg, self.Wfb) # filtering
                HWemg = tf.reshape(HWemg, [-1, self.config.epoch_seq_len, self.config.seq_frame_seq_len, self.config.seq_nfilter])

        if(self.config.nchannel > 2):
            X2 = tf.concat([HWeeg, HWeog, HWemg], axis = 3)
        elif(self.config.nchannel > 1):
            X2 = tf.concat([HWeeg, HWeog], axis = 3)
        else:
            X2 = HWeeg
        X2 = tf.reshape(X2, [-1, self.config.seq_frame_seq_len, self.config.seq_nfilter*self.config.nchannel])

        # bidirectional epoch-level recurrent layer
        with tf.variable_scope("seq_frame_rnn_layer") as scope:
            seq_fw_cell1, seq_bw_cell1 = bidirectional_recurrent_layer_bn(self.config.seq_nhidden1,
                                                                  self.config.seq_nlayer1,
                                                                  seq_len=self.config.seq_frame_seq_len,
                                                                  is_training=self.istraining,
                                                                  input_keep_prob=self.dropout_rnn, # we have dropouted in the convolutional layer
                                                                  output_keep_prob=self.dropout_rnn)
            seq_rnn_out1, seq_rnn_state1 = bidirectional_recurrent_layer_output(seq_fw_cell1,
                                                                                    seq_bw_cell1,
                                                                                    X2,
                                                                                    self.seq_frame_seq_len,
                                                                                    scope=scope)
            print(seq_rnn_out1.get_shape())

        with tf.variable_scope("seq_frame_attention_layer"):
            self.seq_attention_out1, _ = attention(seq_rnn_out1, self.config.seq_attention_size1)
            print(self.seq_attention_out1.get_shape())

        seq_e_rnn_input = tf.reshape(self.seq_attention_out1, [-1, self.config.epoch_seq_len, self.config.seq_nhidden1*2])
        # bidirectional sequence-level recurrent layer
        with tf.variable_scope("seq_epoch_rnn_layer") as scope:
            seq_fw_cell2, seq_bw_cell2 = bidirectional_recurrent_layer_bn(self.config.seq_nhidden2,
                                                                  self.config.seq_nlayer2,
                                                                  seq_len=self.config.epoch_seq_len,
                                                                  is_training=self.istraining,
                                                                  input_keep_prob=self.dropout_rnn, # we have dropouted the output of frame-wise rnn
                                                                  output_keep_prob=self.dropout_rnn)
            self.seq_rnn_out2, self.seq_rnn_state2 = bidirectional_recurrent_layer_output(seq_fw_cell2,
                                                                             seq_bw_cell2,
                                                                             seq_e_rnn_input,
                                                                             self.epoch_seq_len,
                                                                             scope=scope)
            print(self.seq_rnn_out2.get_shape())

        self.seq_scores = []
        self.seq_predictions = []
        with tf.variable_scope("seq_output_layer"):
            for i in range(self.config.epoch_seq_len):
                seq_score_i = fc(tf.squeeze(self.seq_rnn_out2[:,i,:]),
                                self.config.seq_nhidden2 * 2,
                                self.config.nclass,
                                name="seq_output",
                                relu=False)
                seq_pred_i = tf.argmax(seq_score_i, 1, name="pred-%s" % i)
                self.seq_scores.append(seq_score_i)
                self.seq_predictions.append(seq_pred_i)


        self.seq_loss = 0
        with tf.name_scope("seq-loss"):
            for i in range(self.config.epoch_seq_len):
                seq_loss_i = tf.nn.softmax_cross_entropy_with_logits(labels=tf.squeeze(self.input_y[:,i,:]), logits=self.seq_scores[i])
                seq_loss_i = tf.reduce_sum(seq_loss_i, axis=[0])
                self.seq_loss += seq_loss_i
        self.seq_loss = self.seq_loss/self.config.epoch_seq_len # average over sequence length

        self.seq_accuracy = []
        # Accuracy
        with tf.name_scope("seq-accuracy"):
            for i in range(self.config.epoch_seq_len):
                seq_correct_prediction_i = tf.equal(self.seq_predictions[i], tf.argmax(tf.squeeze(self.input_y[:,i,:]), 1))
                seq_accuracy_i = tf.reduce_mean(tf.cast(seq_correct_prediction_i, "float"), name="seq-accuracy-%s" % i)
                self.seq_accuracy.append(seq_accuracy_i)
        #############################################################
        # End seqsleepnet
        #############################################################


    #############################################################
    # Joint branch
    #############################################################
    def construct_joint_branch(self):

        self.joint_scores = []
        self.joint_predictions = []
        with tf.variable_scope("joint_output_layer"):
            for i in range(self.config.epoch_seq_len):
                joint_score_i = fc(tf.concat([tf.squeeze(self.deep_rnn_out[:,i,:]), tf.squeeze(self.seq_rnn_out2[:,i,:])], 1),
                                self.config.seq_nhidden2 * 2 + self.config.deep_nhidden * 2,
                                self.config.nclass,
                                name="joint_output",
                                relu=False)
                joint_pred_i = tf.argmax(joint_score_i, 1, name="pred-%s" % i)
                self.joint_scores.append(joint_score_i)
                self.joint_predictions.append(joint_pred_i)

        self.joint_loss = 0
        with tf.name_scope("joint-loss"):
            for i in range(self.config.epoch_seq_len):
                joint_loss_i = tf.nn.softmax_cross_entropy_with_logits(labels=tf.squeeze(self.input_y[:,i,:]), logits=self.joint_scores[i])
                joint_loss_i = tf.reduce_sum(joint_loss_i, axis=[0])
                self.joint_loss += joint_loss_i
        self.joint_loss = self.joint_loss/self.config.epoch_seq_len # average over sequence length

        self.joint_accuracy = []
        # Accuracy
        with tf.name_scope("joint-accuracy"):
            for i in range(self.config.epoch_seq_len):
                joint_correct_prediction_i = tf.equal(self.joint_predictions[i], tf.argmax(tf.squeeze(self.input_y[:,i,:]), 1))
                joint_accuracy_i = tf.reduce_mean(tf.cast(joint_correct_prediction_i, "float"), name="joint-accuracy-%s" % i)
                self.joint_accuracy.append(joint_accuracy_i)
        #############################################################
        # End joint branch
        #############################################################


    def all_convolution_block(self, input, name):
        in_dims = input.get_shape().as_list()
        print(in_dims)
        h_i = input
        if len(in_dims) == 2:
            h_i = tf.expand_dims(input, -1)
        elif len(in_dims) < 2 or len(in_dims) > 3:
            raise ValueError('Generator input must be 2-D or 3-D')
        kwidth = 31

        with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
            for layer_idx, layer_depth in enumerate(self.g_enc_depths):
                bias_init = tf.constant_initializer(0.)
                h_i_dwn = downconv(h_i, layer_depth, kwidth=kwidth,
                                   init=tf.truncated_normal_initializer(stddev=0.02),
                                   bias_init=bias_init,
                                   name='enc_{}'.format(layer_idx))
                print("h_i_dwn")
                print(h_i_dwn.get_shape())
                print('Downconv {} -> {}'.format(h_i.get_shape(),h_i_dwn.get_shape()))
                h_i = h_i_dwn
                print('-- Enc: leakyrelu activation --')
                h_i = leakyrelu(h_i)
                if(layer_idx < len(self.g_enc_depths) - 1):
                    h_i = dropout(h_i, self.dropout_cnn)
        return h_i
