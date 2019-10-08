class Config(object):
    def __init__(self):

        # common settings
        self.epoch_seq_len = 20 # seq_len
        self.nchannel = 3       # number of channels
        self.nclass = 5         # number of sleep stages

        self.learning_rate = 1e-4
        self.l2_reg_lambda = 0.001
        self.training_epoch = 10
        self.batch_size = 32
        self.evaluate_every = 200  # evaluate and save checkpoints after 200 training steps
        self.checkpoint_every = 200

        # settings for the seqsleepnet branch
        self.seq_ndim = 129 # freq dimension
        self.seq_frame_seq_len = 29 # time dimension
        self.seq_dropout_keep_prob_rnn = 0.75 # biRNN drop-out keep rate

        self.seq_nhidden1 = 64          # number of hidden unit for the epoch-wise biRNN
        self.seq_nlayer1 = 1            # number of epoch-wise biRNN layers
        self.seq_attention_size1 = 32   # attention size of the epoch-wise biRNN
        self.seq_nhidden2 = 64          # number of hidden unit for the sequence-wise biRNN
        self.seq_nlayer2 = 1            # number of sequence-wise biRNN layers

        self.seq_nfilter = 32           # number of filters in the filter-bank layer
        self.seq_nfft = 256             # nfft that was used in STFT when computing time-frequency input
        self.seq_samplerate = 100       # sampling rate
        self.seq_lowfreq = 0            # low frequency bound of the filterbank
        self.seq_highfreq = 50          # high frequency bound of the filterbank

        # settings for deepsleepnet branch
        self.deep_nlayer = 2            # number of biRNN on top of the CNNs
        self.deep_ndim = 1              # frequency dimension of the input (1 as the raw input)
        self.deep_ntime = 3000          # time dimension
        self.deep_nhidden = 512         # number of hidden units in the biRNN
        self.deep_nstep = 25            # the number of time steps per series
