class Config(object):
    def __init__(self):

        # common settings
        self.epoch_seq_len = 20 # seq_len
        self.nchannel = 3 # number of channels
        self.nclass = 5

        self.learning_rate = 1e-4
        self.l2_reg_lambda = 0.001
        self.training_epoch = 10
        self.batch_size = 32
        self.evaluate_every = 200
        self.checkpoint_every = 200

        # seqsleepnet settings
        self.seq_ndim = 129 # freq
        self.seq_frame_seq_len = 29 # time

        self.seq_nhidden1 = 64
        self.seq_nlayer1 = 1
        self.seq_attention_size1 = 32
        self.seq_nhidden2 = 64
        self.seq_nlayer2 = 1

        self.seq_nfilter = 32
        self.seq_nfft = 256
        self.seq_samplerate = 100
        self.seq_lowfreq = 0
        self.seq_highfreq = 50

        # fcnnrnn settings
        self.deep_nlayer = 2
        self.deep_ntime = 3000  # time dimension
        self.deep_nhidden = 256  # nb of neurons in the hidden layer of the GRU cell

        self.dropout_rnn = 0.75
        self.dropout_cnn = 0.5
