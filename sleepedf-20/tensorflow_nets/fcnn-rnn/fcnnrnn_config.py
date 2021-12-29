# import tensorflow as tf
import numpy as np
import os


class Config(object):
    def __init__(self):
        # Trainging
        self.learning_rate = 1e-4
        self.l2_reg_lambda = 0.001
        self.training_epoch = 10
        self.batch_size = 32
        # dropout for fully connected layers
        self.dropout_cnn = 0.5
        self.dropout_rnn = 0.75

        self.evaluate_every = 100

        self.nlayer = 2
        self.ndim = 1  # frequency dimension
        self.ntime = 3000  # time dimension
        self.nchannel = 3  # channel dimension
        self.nhidden = 256  # nb of neurons in the hidden layer of the GRU cell
        self.nstep = 20  # the number of time steps per series

        self.nclass = 5  # Final output classes

        self.epoch_seq_len = 20
