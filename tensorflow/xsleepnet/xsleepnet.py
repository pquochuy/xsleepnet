import tensorflow as tf
from nn_basic_layers import *
from filterbank_shape import FilterbankShape

class XSleep_Net(object):

    def __init__(self, config):
        # Placeholders for input, output and dropout
        self.config = config
        # raw input
        self.input_x1 = tf.placeholder(tf.float32,shape=[None, self.config.epoch_seq_len, self.config.deep_ntime, self.config.deep_ndim, self.config.nchannel],name='input_x1')
        # time-frequency input
        self.input_x2 = tf.placeholder(tf.float32, [None, self.config.epoch_seq_len, self.config.seq_frame_seq_len, self.config.seq_ndim, self.config.nchannel], name="input_x2")
        # one-hot encoding
        self.input_y = tf.placeholder(tf.float32, [None, self.config.epoch_seq_len, self.config.nclass], name="input_y")
        # drop-out rate of the seqsleepnet branch
        self.seq_dropout_keep_prob_rnn = tf.placeholder(tf.float32, name="seq_dropout_keep_prob_rnn")
        # drop-out rate of the deep branch
        self.deep_dropout = tf.placeholder(tf.float32, name="deep_dropout")
        self.istraining = tf.placeholder(tf.bool, name='istraining') # idicate training for batch normmalization

        # length of each input sequence of epochs
        self.seq_frame_seq_len = tf.placeholder(tf.int32, [None]) # for the dynamic RNN
        # the number of spectral columns in one time-frequency input
        self.epoch_seq_len = tf.placeholder(tf.int32, [None]) # for the dynamic RNN

        # weight for seqsleepnet branch
        self.w1 = tf.placeholder(tf.float32, name="w1")
        # weight for deepsleepnet branch
        self.w2 = tf.placeholder(tf.float32, name="w2")
        # weight for the joint branch
        self.w3 = tf.placeholder(tf.float32, name="w3")

        #############################################################
        # Construct deepsleepnet branch here
        #############################################################
        X1 = tf.reshape(self.input_x1, [-1, self.config.deep_ntime, self.config.deep_ndim, self.config.nchannel])

        # CNN brach 1
        conv11 = conv_bn(X1, 50, 3, 64, 6, 1, is_training=self.istraining, padding='SAME', name='deep_conv11') # 50 = Fs/2, stride = 6 (according to original implementation)
        print(conv11.get_shape()) # [batchsize x epoch_step, x, 1, 64] (x = (3000-50)/6 + 1 = 492
        pool11 = max_pool(conv11, 8, 1, 8, 1,padding='SAME', name='deep_pool11')
        print(pool11.get_shape())
        # pool1 shape [batchsize x epoch_step, 63, 1, 64] (63 is due to SAME padding above)
        dropout11 = dropout(pool11, self.deep_dropout)

        conv12 = conv_bn(dropout11, 8, 1, 128, 1, 1, is_training=self.istraining, padding='SAME', name='deep_conv12')
        print(conv12.get_shape()) # [batchsize x epoch_step, 63, 1, 128]
        conv13 = conv_bn(conv12, 8, 1, 128, 1, 1, is_training=self.istraining, padding='SAME', name='deep_conv13')
        print(conv13.get_shape()) # [batchsize x epoch_step, 63, 1, 128]
        conv14 = conv_bn(conv13, 8, 1, 128, 1, 1, is_training=self.istraining, padding='SAME', name='deep_conv14')
        print(conv14.get_shape()) # [batchsize x epoch_step, 63, 1, 128]
        pool14 = max_pool(conv14, 4, 1, 4, 1,padding='SAME', name='deep_pool14')
        print(pool14.get_shape()) # [batchsize x epoch_step, 16, 1, 128]
        pool14 = tf.squeeze(pool14) #[batchsize x epoch_step, 16, 128]

        # CNN brach 1
        conv21 = conv_bn(X1, 400, 3, 64, 50, 1, is_training=self.istraining, padding='SAME', name='deep_conv21') # 400 = Fsx4, 50 = Fs/2
        print(conv21.get_shape()) # [batchsize x epoch_step, x, 1, 64] (x = (3000-400)/50 + 1 = 53
        pool21 = max_pool(conv21, 4, 1, 4, 1,padding='SAME', name='deep_pool21')
        print(pool21.get_shape()) # [batchsize x epoch_step, 15, 1, 64] # 14 is due to SAME padding above
        dropout21 = dropout(pool21, self.deep_dropout)

        conv22 = conv_bn(dropout21, 6, 1, 128, 1, 1, is_training=self.istraining, padding='SAME', name='deep_conv22')
        print(conv22.get_shape()) # [batchsize x epoch_step, 15, 1, 128]
        conv23 = conv_bn(conv22, 6, 1, 128, 1, 1, is_training=self.istraining, padding='SAME', name='deep_conv23')
        print(conv23.get_shape()) # [batchsize x epoch_step, 15, 1, 128]
        conv24 = conv_bn(conv23, 6, 1, 128, 1, 1, is_training=self.istraining, padding='SAME', name='deep_conv24')
        print(conv24.get_shape()) # [batchsize x epoch_step, 15, 1, 128]
        pool24 = max_pool(conv24, 2, 1, 2, 1,padding='SAME', name='deep_pool14')
        print(pool24.get_shape()) # [batchsize x epoch_step, 8, 1, 128]
        pool24 = tf.squeeze(pool24) # [batchsize x epoch_step, 8, 128]

        # concatenate
        cnn_concat = tf.concat([pool14, pool24], axis = 1) # [batchsize x epoch_step, 24, 128]
        cnn_output = tf.reshape(cnn_concat, [-1, 24*128]) #[batchsize x epoch_step, 24*128]
        cnn_output = dropout(cnn_output, self.deep_dropout)
        # residual
        residual_output = fc_bn(cnn_output, 24*128, 1024, is_training=self.istraining, name='deep_residual_layer', relu=True)
        #[batchsize x epoch_step, 1024]
        residual_output = tf.reshape(residual_output, [-1, self.config.epoch_seq_len, 1024]) #[batchsize x epoch_step, 32*128]
        print(residual_output.get_shape())

        deep_rnn_input = tf.reshape(cnn_concat, [-1, self.config.epoch_seq_len, 24*128])

        # bidirectional recurrent layers of deepsleepnet
        with tf.device('/gpu:0'), tf.variable_scope("deep_epoch_rnn_layer") as scope:
            deep_fw_cell, deep_bw_cell = bidirectional_recurrent_layer_lstm_new(self.config.deep_nhidden,
                                                                  self.config.deep_nlayer,
                                                                  input_keep_prob=self.deep_dropout,
                                                                  output_keep_prob=self.deep_dropout)
            deep_rnn_out, deep_rnn_state = bidirectional_recurrent_layer_output_new(deep_fw_cell,
                                                                      deep_bw_cell,
                                                                      deep_rnn_input,
                                                                      self.epoch_seq_len,
                                                                      scope=scope)
            print(deep_rnn_out.get_shape())

        # joint for final output
        deep_final_output = tf.add(deep_rnn_out, residual_output)
        deep_final_output = dropout(deep_final_output, self.deep_dropout)

        # output layers of deepsleepnet branch
        self.deep_scores = []
        self.deep_predictions = []
        with tf.device('/gpu:0'), tf.variable_scope("deep_output_layer"):
            for i in range(self.config.epoch_seq_len):
                deep_score_i = fc(tf.squeeze(deep_final_output[:,i,:]),
                                self.config.deep_nhidden * 2,
                                self.config.nclass,
                                name="deep_output", # same variable scope to force weight sharing
                                relu=False)
                deep_pred_i = tf.argmax(deep_score_i, 1, name="deep_pred-%s" % i)
                self.deep_scores.append(deep_score_i)
                # predictions of this branch
                self.deep_predictions.append(deep_pred_i)

        # loss of deepsleepnet branch
        self.output_loss1 = 0
        with tf.device('/gpu:0'), tf.name_scope("output-loss1"):
            for i in range(self.config.epoch_seq_len):
                loss1_i = tf.nn.softmax_cross_entropy_with_logits(labels=tf.squeeze(self.input_y[:,i,:]), logits=self.deep_scores[i])
                loss1_i = tf.reduce_sum(loss1_i, axis=[0])
                self.output_loss1 += loss1_i
        self.output_loss1 = self.output_loss1/self.config.epoch_seq_len # average over sequence length

        # accuracy of of deepsleepnet branch
        self.accuracy1 = []
        # Accuracy
        with tf.device('/gpu:0'), tf.name_scope("accuracy1"):
            for i in range(self.config.epoch_seq_len):
                correct_prediction1_i = tf.equal(self.deep_predictions[i], tf.argmax(tf.squeeze(self.input_y[:,i,:]), 1))
                accuracy1_i = tf.reduce_mean(tf.cast(correct_prediction1_i, "float"), name="accuracy-%s" % i)
                self.accuracy1.append(accuracy1_i)
        #############################################################
        # End deepsleepnet here
        #############################################################


        #############################################################
        # Construct seqsleepnet branch here
        #############################################################
        filtershape = FilterbankShape()
        #triangular filterbank
        self.Wbl = tf.constant(filtershape.lin_tri_filter_shape(nfilt=self.config.seq_nfilter,
                                                                nfft=self.config.seq_nfft,
                                                                samplerate=self.config.seq_samplerate,
                                                                lowfreq=self.config.seq_lowfreq,
                                                                highfreq=self.config.seq_highfreq),
                               dtype=tf.float32,
                               name="W-filter-shape-eeg")

        # EEG filterbank layer
        with tf.device('/gpu:0'), tf.variable_scope("seq_filterbank-layer-eeg"):
            # Temporarily crush the feature_mat's dimensions
            Xeeg = tf.reshape(tf.squeeze(self.input_x2[:,:,:,:,0]), [-1, self.config.seq_ndim])
            # first filter bank layer
            self.Weeg = tf.Variable(tf.random_normal([self.config.seq_ndim, self.config.seq_nfilter],dtype=tf.float32))
            # non-negative constraints
            self.Weeg = tf.sigmoid(self.Weeg)
            # mask matrix should be replaced by shape-specific filter bank
            self.Wfb = tf.multiply(self.Weeg,self.Wbl)
            HWeeg = tf.matmul(Xeeg, self.Wfb) # filtering
            HWeeg = tf.reshape(HWeeg, [-1, self.config.epoch_seq_len, self.config.seq_frame_seq_len, self.config.seq_nfilter])

        # EOG filterbank layer
        if(self.config.nchannel > 1):
            with tf.device('/gpu:0'), tf.variable_scope("seq_filterbank-layer-eog"):
                # Temporarily crush the feature_mat's dimensions
                Xeog = tf.reshape(tf.squeeze(self.input_x2[:,:,:,:,1]), [-1, self.config.seq_ndim])
                # first filter bank layer
                self.Weog = tf.Variable(tf.random_normal([self.config.seq_ndim, self.config.seq_nfilter],dtype=tf.float32))
                # non-negative constraints
                self.Weog = tf.sigmoid(self.Weog)
                # mask matrix should be replaced by shape-specific filter bank
                self.Wfb = tf.multiply(self.Weog,self.Wbl)
                HWeog = tf.matmul(Xeog, self.Wfb) # filtering
                HWeog = tf.reshape(HWeog, [-1, self.config.epoch_seq_len, self.config.seq_frame_seq_len, self.config.seq_nfilter])

        # EMG filterbank layer
        if(self.config.nchannel > 2):
            with tf.device('/gpu:0'), tf.variable_scope("seq_filterbank-layer-emg"):
                # Temporarily crush the feature_mat's dimensions
                Xemg = tf.reshape(tf.squeeze(self.input_x2[:,:,:,:,2]), [-1, self.config.seq_ndim])
                # first filter bank layer
                self.Wemg = tf.Variable(tf.random_normal([self.config.seq_ndim, self.config.seq_nfilter],dtype=tf.float32))
                # non-negative constraints
                self.Wemg = tf.sigmoid(self.Wemg)
                # mask matrix should be replaced by shape-specific filter bank
                self.Wfb = tf.multiply(self.Wemg,self.Wbl)
                HWemg = tf.matmul(Xemg, self.Wfb) # filtering
                HWemg = tf.reshape(HWemg, [-1, self.config.epoch_seq_len, self.config.seq_frame_seq_len, self.config.seq_nfilter])

        # concatenate for 3-channel input
        if(self.config.nchannel > 2):
            X2 = tf.concat([HWeeg, HWeog, HWemg], axis = 3)
        # concatenate for 2-channel input
        elif(self.config.nchannel > 1):
            X2 = tf.concat([HWeeg, HWeog], axis = 3)
        # otherwise, only EEG
        else:
            X2 = HWeeg
        X2 = tf.reshape(X2, [-1, self.config.seq_frame_seq_len, self.config.seq_nfilter*self.config.nchannel])

        # epoch-wise biRNN layer
        with tf.device('/gpu:0'), tf.variable_scope("seq_frame_rnn_layer") as scope:
            seq_fw_cell1, seq_bw_cell1 = bidirectional_recurrent_layer_bn_new(self.config.seq_nhidden1,
                                                                  self.config.seq_nlayer1,
                                                                  seq_len=self.config.seq_frame_seq_len,
                                                                  is_training=self.istraining,
                                                                  input_keep_prob=self.seq_dropout_keep_prob_rnn,
                                                                  output_keep_prob=self.seq_dropout_keep_prob_rnn)
            seq_rnn_out1, seq_rnn_state1 = bidirectional_recurrent_layer_output_new(seq_fw_cell1,
                                                                                    seq_bw_cell1,
                                                                                    X2,
                                                                                    self.seq_frame_seq_len,
                                                                                    scope=scope)
            print(seq_rnn_out1.get_shape())
            # output shape (batchsize*epoch_step, frame_step, nhidden1*2)

        # attention layer for pooling
        with tf.device('/gpu:0'), tf.variable_scope("seq_frame_attention_layer"):
            self.seq_attention_out1, _ = attention(seq_rnn_out1, self.config.seq_attention_size1)
            print(self.seq_attention_out1.get_shape())
            # attention_output1 of shape (batchsize*epoch_step, nhidden1*2)

        # sequence-wise biRNN layer
        seq_e_rnn_input = tf.reshape(self.seq_attention_out1, [-1, self.config.epoch_seq_len, self.config.seq_nhidden1*2])
        with tf.device('/gpu:0'), tf.variable_scope("seq_epoch_rnn_layer") as scope:
            seq_fw_cell2, seq_bw_cell2 = bidirectional_recurrent_layer_bn_new(self.config.seq_nhidden2,
                                                                  self.config.seq_nlayer2,
                                                                  seq_len=self.config.epoch_seq_len,
                                                                  is_training=self.istraining,
                                                                  input_keep_prob=self.seq_dropout_keep_prob_rnn,
                                                                  output_keep_prob=self.seq_dropout_keep_prob_rnn)
            seq_rnn_out2, seq_rnn_state2 = bidirectional_recurrent_layer_output_new(seq_fw_cell2,
                                                                             seq_bw_cell2,
                                                                             seq_e_rnn_input,
                                                                             self.epoch_seq_len,
                                                                             scope=scope)
            print(seq_rnn_out2.get_shape())
            # output2 of shape (batchsize, epoch_step, nhidden2*2)

        # output layer of seqsleepnet branch
        self.seq_scores = []
        self.seq_predictions = []
        with tf.device('/gpu:0'), tf.variable_scope("seq_output_layer"):
            for i in range(self.config.epoch_seq_len):
                seq_score_i = fc(tf.squeeze(seq_rnn_out2[:,i,:]),
                                self.config.seq_nhidden2 * 2,
                                self.config.nclass,
                                name="seq_output", # same variable scope to force weight sharing
                                relu=False)
                seq_pred_i = tf.argmax(seq_score_i, 1, name="pred-%s" % i)
                self.seq_scores.append(seq_score_i)
                self.seq_predictions.append(seq_pred_i)

        # loss of seqsleepnet branch
        self.output_loss2 = 0
        with tf.device('/gpu:0'), tf.name_scope("output-loss2"):
            for i in range(self.config.epoch_seq_len):
                loss2_i = tf.nn.softmax_cross_entropy_with_logits(labels=tf.squeeze(self.input_y[:,i,:]), logits=self.seq_scores[i])
                loss2_i = tf.reduce_sum(loss2_i, axis=[0])
                self.output_loss2 += loss2_i
        self.output_loss2 = self.output_loss2/self.config.epoch_seq_len # average over sequence length

        self.accuracy2 = []
        # Accuracy
        with tf.device('/gpu:0'), tf.name_scope("accuracy2"):
            for i in range(self.config.epoch_seq_len):
                correct_prediction2_i = tf.equal(self.seq_predictions[i], tf.argmax(tf.squeeze(self.input_y[:,i,:]), 1))
                accuracy2_i = tf.reduce_mean(tf.cast(correct_prediction2_i, "float"), name="accuracy-%s" % i)
                self.accuracy2.append(accuracy2_i)
        #############################################################
        # End seqsleepnet branch
        #############################################################


        #############################################################
        # This is the joint branch that joins seqsleepnet and deepsleepnet outputs
        #############################################################
        # output layer of the joint branch
        self.joint_scores = []
        self.joint_predictions = []
        with tf.device('/gpu:0'), tf.variable_scope("joint_output_layer"):
            for i in range(self.config.epoch_seq_len):
                # concatenate seqsleepnet and deepsleepnet outputs
                joint_score_i = fc(tf.concat([tf.squeeze(seq_rnn_out2[:,i,:]), tf.squeeze(deep_final_output[:,i,:])], 1),
                                self.config.seq_nhidden2 * 2 + self.config.deep_nhidden * 2,
                                self.config.nclass,
                                name="joint_output",
                                relu=False)
                joint_pred_i = tf.argmax(joint_score_i, 1, name="pred-%s" % i)
                self.joint_scores.append(joint_score_i)
                self.joint_predictions.append(joint_pred_i)

        # loss of the joint branch
        self.output_loss3 = 0
        with tf.device('/gpu:0'), tf.name_scope("output-loss3"):
            for i in range(self.config.epoch_seq_len):
                loss3_i = tf.nn.softmax_cross_entropy_with_logits(labels=tf.squeeze(self.input_y[:,i,:]), logits=self.joint_scores[i])
                loss3_i = tf.reduce_sum(loss3_i, axis=[0])
                self.output_loss3 += loss3_i
        self.output_loss3 = self.output_loss3/self.config.epoch_seq_len # average over sequence length

        self.accuracy3 = []
        # Accuracy
        with tf.device('/gpu:0'), tf.name_scope("accuracy3"):
            for i in range(self.config.epoch_seq_len):
                correct_prediction3_i = tf.equal(self.joint_predictions[i], tf.argmax(tf.squeeze(self.input_y[:,i,:]), 1))
                accuracy3_i = tf.reduce_mean(tf.cast(correct_prediction3_i, "float"), name="accuracy-%s" % i)
                self.accuracy3.append(accuracy3_i)
        #############################################################
        # End of the joint branch
        #############################################################


        # Total weighting loss of all 3 branches
        self.output_loss = self.w1*self.output_loss1 + self.w2*self.output_loss2 + self.w3*self.output_loss3

        # add on regularization except the seqsleepnet branch's filterbank layers
        with tf.device('/gpu:0'), tf.name_scope("l2_loss"):
            vars   = tf.trainable_variables()
            except_vars_eeg = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='seq_filterbank-layer-eeg')
            except_vars_eog = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='seq_filterbank-layer-eog')
            except_vars_emg = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='seq_filterbank-layer-emg')
            l2_loss = tf.add_n([ tf.nn.l2_loss(v) for v in vars
                    if v not in except_vars_eeg and v not in except_vars_eog and v not in except_vars_emg])
            self.loss = self.output_loss + self.config.l2_reg_lambda*l2_loss


        # final output and accuracy
        self.accuracy = []
        self.score = []
        self.predictions = []
        # Accuracy
        with tf.device('/gpu:0'), tf.name_scope("accuracy"):
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
