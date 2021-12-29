import tensorflow as tf

from nn_basic_layers import *
from ops import *

import numpy as np
import os


class FCNNRNN(object):
    def __init__(self, config, is_eog=True, is_emg=True):
        self.g_enc_depths = [16, 16, 32, 32, 64, 64, 128, 128, 256]
        self.d_num_fmaps = [16, 16, 32, 32, 64, 64, 128, 128, 256]
        # Placeholders for input, output and dropout
        self.config = config
        self.is_emg = is_emg
        self.is_eog = is_eog
        self.input_x = tf.placeholder(tf.float32,shape=[None, self.config.epoch_step, self.config.ntime, self.config.nchannel],name='input_x')
        self.input_y = tf.placeholder(tf.float32, shape=[None, self.config.epoch_step, self.config.nclass], name='input_y')
        self.dropout_cnn = tf.placeholder(tf.float32, name="dropout_cnn")
        self.dropout_rnn = tf.placeholder(tf.float32, name="dropout_rnn")

        self.istraining = tf.placeholder(tf.bool, name='istraining') # indicate training for batch normmalization

        self.epoch_seq_len = tf.placeholder(tf.int32, [None]) # for the dynamic RNN
        X = tf.reshape(self.input_x, [-1, self.config.ntime, self.config.nchannel])

        conv_feat = self.all_convolution_block(X,"conv_eeg")
        Nfeat = 6*self.g_enc_depths[-1]
        conv_feat = tf.reshape(conv_feat, [-1, Nfeat])

        print("conv_feat")
        print(conv_feat.get_shape())

        rnn_input = tf.reshape(conv_feat, [-1, self.config.epoch_seq_len, Nfeat])

        with tf.variable_scope("epoch_rnn_layer") as scope:
            fw_cell, bw_cell = bidirectional_recurrent_layer(self.config.nhidden,
                                                                  self.config.nlayer,
                                                                  input_keep_prob=self.dropout_rnn,
                                                                  output_keep_prob=self.dropout_rnn)
            rnn_out, rnn_state = bidirectional_recurrent_layer_output(fw_cell,
                                                                      bw_cell,
                                                                      rnn_input,
                                                                      self.epoch_seq_len,
                                                                      scope=scope)
            print(rnn_out.get_shape())


        self.scores = []
        self.predictions = []
        with tf.variable_scope("output_layer"):
            for i in range(self.config.epoch_step):
                score_i = fc(tf.squeeze(rnn_out[:,i,:]),
                                self.config.nhidden * 2,
                                self.config.nclass,
                                name="output",
                                relu=False)
                pred_i = tf.argmax(score_i, 1, name="pred-%s" % i)
                self.scores.append(score_i)
                self.predictions.append(pred_i)

        # calculate cross-entropy loss
        self.output_loss = 0
        with tf.name_scope("output-loss"):
            for i in range(self.config.epoch_step):
                output_loss_i = tf.nn.softmax_cross_entropy_with_logits(labels=tf.squeeze(self.input_y[:,i,:]), logits=self.scores[i])
                output_loss_i = tf.reduce_sum(output_loss_i, axis=[0])
                self.output_loss += output_loss_i
        self.output_loss = self.output_loss/self.config.epoch_step

        # add on regularization
        with tf.name_scope("l2_loss"):
            vars   = tf.trainable_variables()
            l2_loss = tf.add_n([ tf.nn.l2_loss(v) for v in vars])
            self.loss = self.output_loss + self.config.l2_reg_lambda*l2_loss

        self.accuracy = []
        # Accuracy
        with tf.name_scope("accuracy"):
            for i in range(self.config.epoch_step):
                correct_prediction_i = tf.equal(self.predictions[i], tf.argmax(tf.squeeze(self.input_y[:,i,:]), 1))
                accuracy_i = tf.reduce_mean(tf.cast(correct_prediction_i, "float"), name="accuracy-%s" % i)
                self.accuracy.append(accuracy_i)


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
