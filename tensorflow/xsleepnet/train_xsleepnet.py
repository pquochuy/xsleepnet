import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0,-1"
import numpy as np
import tensorflow as tf

#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())

import shutil, sys
from datetime import datetime
import h5py

from xsleepnet import XSleep_Net
from xsleepnet_config import Config

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score

from datagenerator_from_list_v2 import DataGenerator

from scipy.io import loadmat

import copy

import time

# Parameters
# ==================================================

# Misc Parameters
tf.app.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.app.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# My Parameters
tf.app.flags.DEFINE_string("eeg_train_data", "../train_data.mat", "Point to directory of input data")
tf.app.flags.DEFINE_string("eeg_train_data_check", "../train_data.mat", "Point to directory of input data")
tf.app.flags.DEFINE_string("eeg_eval_data", "../data/eval_data_1.mat", "Point to directory of input data")
tf.app.flags.DEFINE_string("eeg_test_data", "../test_data.mat", "Point to directory of input data")
tf.app.flags.DEFINE_string("eog_train_data", "../train_data.mat", "Point to directory of input data")
tf.app.flags.DEFINE_string("eog_train_data_check", "../train_data.mat", "Point to directory of input data")
tf.app.flags.DEFINE_string("eog_eval_data", "../data/eval_data_1.mat", "Point to directory of input data")
tf.app.flags.DEFINE_string("eog_test_data", "../test_data.mat", "Point to directory of input data")
tf.app.flags.DEFINE_string("emg_train_data", "../train_data.mat", "Point to directory of input data")
tf.app.flags.DEFINE_string("emg_train_data_check", "../train_data.mat", "Point to directory of input data")
tf.app.flags.DEFINE_string("emg_eval_data", "../data/eval_data_1.mat", "Point to directory of input data")
tf.app.flags.DEFINE_string("emg_test_data", "../test_data.mat", "Point to directory of input data")
tf.app.flags.DEFINE_string("out_dir", "./output/", "Point to output directory")
tf.app.flags.DEFINE_string("checkpoint_dir", "./checkpoint/", "Point to checkpoint directory")

# seqsleepnet settings
tf.app.flags.DEFINE_float("seq_dropout_keep_prob_rnn", 0.75, "Dropout keep probability (default: 0.75)")
tf.app.flags.DEFINE_integer("seq_nfilter", 32, "Sequence length (default: 20)")
tf.app.flags.DEFINE_integer("seq_nhidden1", 64, "Sequence length (default: 20)")
tf.app.flags.DEFINE_integer("seq_attention_size1", 32, "Sequence length (default: 20)")
tf.app.flags.DEFINE_integer("seq_nhidden2", 64, "Sequence length (default: 20)")

# deepsleepnet settings
tf.app.flags.DEFINE_float("deep_dropout", 0.5, "Dropout keep probability (default: 0.75)")
tf.app.flags.DEFINE_integer("deep_nhidden", 512, "Sequence length (default: 20)")

# common settings
tf.app.flags.DEFINE_integer("seq_len", 20, "Sequence length (default: 32)")


FLAGS = tf.app.flags.FLAGS
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()): # python3
    print("{}={}".format(attr.upper(), value))
print("")

# Data Preparatopn
# ==================================================

# path where some output are stored
out_path = os.path.abspath(os.path.join(os.path.curdir,FLAGS.out_dir))
# path where checkpoint models are stored
checkpoint_path = os.path.abspath(os.path.join(out_path,FLAGS.checkpoint_dir))
if not os.path.isdir(os.path.abspath(out_path)): os.makedirs(os.path.abspath(out_path))
if not os.path.isdir(os.path.abspath(checkpoint_path)): os.makedirs(os.path.abspath(checkpoint_path))

config = Config()
config.seq_dropout_keep_prob_rnn = FLAGS.seq_dropout_keep_prob_rnn
config.epoch_seq_len = FLAGS.seq_len
config.seq_nfilter = FLAGS.seq_nfilter
config.seq_nhidden1 = FLAGS.seq_nhidden1
config.seq_nhidden2 = FLAGS.seq_nhidden2
config.seq_attention_size1 = FLAGS.seq_attention_size1

config.deep_dropout = FLAGS.deep_dropout
#config.epoch_seq_len = FLAGS.seq_len
#config.epoch_step = FLAGS.seq_len
config.deep_nhidden = FLAGS.deep_nhidden

eeg_active = ((FLAGS.eeg_train_data != "") and (FLAGS.eeg_test_data != ""))
eog_active = ((FLAGS.eog_train_data != "") and (FLAGS.eog_test_data != ""))
emg_active = ((FLAGS.emg_train_data != "") and (FLAGS.emg_test_data != ""))

if (eeg_active):
    print("eeg active")
    # Initalize the EEG data generator seperately for the training, training_check, validation, and test sets
    eeg_train_gen = DataGenerator(os.path.abspath(FLAGS.eeg_train_data),
                                  data_shape_1=[config.deep_ntime],
                                  data_shape_2=[config.seq_frame_seq_len, config.seq_ndim],
                                  seq_len=config.epoch_seq_len, shuffle = False)
    eeg_train_gen_check = DataGenerator(os.path.abspath(FLAGS.eeg_train_data_check),
                                  data_shape_1=[config.deep_ntime],
                                  data_shape_2=[config.seq_frame_seq_len, config.seq_ndim],
                                  seq_len=config.epoch_seq_len, shuffle = False)
    eeg_test_gen = DataGenerator(os.path.abspath(FLAGS.eeg_test_data),
                                 data_shape_1=[config.deep_ntime],
                                 data_shape_2=[config.seq_frame_seq_len, config.seq_ndim],
                                 seq_len=config.epoch_seq_len, shuffle = False)
    eeg_eval_gen = DataGenerator(os.path.abspath(FLAGS.eeg_eval_data),
                                 data_shape_1=[config.deep_ntime],
                                 data_shape_2=[config.seq_frame_seq_len, config.seq_ndim],
                                 seq_len=config.epoch_seq_len, shuffle = False)
    eeg_train_gen.X1 = np.expand_dims(eeg_train_gen.X1, axis=-1) # expand feature dimension
    eeg_train_gen_check.X1 = np.expand_dims(eeg_train_gen_check.X1, axis=-1) # expand feature dimension
    eeg_test_gen.X1 = np.expand_dims(eeg_test_gen.X1, axis=-1) # expand feature dimension
    eeg_eval_gen.X1 = np.expand_dims(eeg_eval_gen.X1, axis=-1) # expand feature dimension

    # data normalization for time-frequency input here
    X2 = eeg_train_gen.X2
    X2 = np.reshape(X2,(eeg_train_gen.data_size*eeg_train_gen.data_shape_2[0], eeg_train_gen.data_shape_2[1]))
    meanX = X2.mean(axis=0)
    stdX = X2.std(axis=0)
    X2 = (X2 - meanX) / stdX
    eeg_train_gen.X2 = np.reshape(X2, (eeg_train_gen.data_size, eeg_train_gen.data_shape_2[0], eeg_train_gen.data_shape_2[1]))

    X2 = eeg_train_gen_check.X2
    X2 = np.reshape(X2,(eeg_train_gen_check.data_size*eeg_train_gen_check.data_shape_2[0], eeg_train_gen_check.data_shape_2[1]))
    X2 = (X2 - meanX) / stdX
    eeg_train_gen_check.X2 = np.reshape(X2, (eeg_train_gen_check.data_size, eeg_train_gen_check.data_shape_2[0], eeg_train_gen_check.data_shape_2[1]))

    X2 = eeg_eval_gen.X2
    X2 = np.reshape(X2,(eeg_eval_gen.data_size*eeg_eval_gen.data_shape_2[0], eeg_eval_gen.data_shape_2[1]))
    X2 = (X2 - meanX) / stdX
    eeg_eval_gen.X2 = np.reshape(X2, (eeg_eval_gen.data_size, eeg_eval_gen.data_shape_2[0], eeg_eval_gen.data_shape_2[1]))

    X2 = eeg_test_gen.X2
    X2 = np.reshape(X2,(eeg_test_gen.data_size*eeg_test_gen.data_shape_2[0], eeg_test_gen.data_shape_2[1]))
    X2 = (X2 - meanX) / stdX
    eeg_test_gen.X2 = np.reshape(X2, (eeg_test_gen.data_size, eeg_test_gen.data_shape_2[0], eeg_test_gen.data_shape_2[1]))

if (eog_active):
    print("eog active")
    # Initalize the EOG data generator seperately for the training, validation, and test sets
    eog_train_gen = DataGenerator(os.path.abspath(FLAGS.eog_train_data),
                                  data_shape_1=[config.deep_ntime],
                                  data_shape_2=[config.seq_frame_seq_len, config.seq_ndim],
                                  seq_len=config.epoch_seq_len, shuffle = False)
    eog_train_gen_check = DataGenerator(os.path.abspath(FLAGS.eog_train_data_check),
                                  data_shape_1=[config.deep_ntime],
                                  data_shape_2=[config.seq_frame_seq_len, config.seq_ndim],
                                  seq_len=config.epoch_seq_len, shuffle = False)
    eog_test_gen = DataGenerator(os.path.abspath(FLAGS.eog_test_data),
                                 data_shape_1=[config.deep_ntime],
                                 data_shape_2=[config.seq_frame_seq_len, config.seq_ndim],
                                 seq_len=config.epoch_seq_len, shuffle = False)
    eog_eval_gen = DataGenerator(os.path.abspath(FLAGS.eog_eval_data),
                                 data_shape_1=[config.deep_ntime],
                                 data_shape_2=[config.seq_frame_seq_len, config.seq_ndim],
                                 seq_len=config.epoch_seq_len, shuffle = False)

    eog_train_gen.X1 = np.expand_dims(eog_train_gen.X1, axis=-1) # expand feature dimension
    eog_train_gen_check.X1 = np.expand_dims(eog_train_gen_check.X1, axis=-1) # expand feature dimension
    eog_test_gen.X1 = np.expand_dims(eog_test_gen.X1, axis=-1) # expand feature dimension
    eog_eval_gen.X1 = np.expand_dims(eog_eval_gen.X1, axis=-1) # expand feature dimension

    # data normalization for time-frequency input
    X2 = eog_train_gen.X2
    X2 = np.reshape(X2,(eog_train_gen.data_size*eog_train_gen.data_shape_2[0], eog_train_gen.data_shape_2[1]))
    meanX = X2.mean(axis=0)
    stdX = X2.std(axis=0)
    X2 = (X2 - meanX) / stdX
    eog_train_gen.X2 = np.reshape(X2, (eog_train_gen.data_size, eog_train_gen.data_shape_2[0], eog_train_gen.data_shape_2[1]))

    X2 = eog_train_gen_check.X2
    X2 = np.reshape(X2,(eog_train_gen_check.data_size*eog_train_gen_check.data_shape_2[0], eog_train_gen_check.data_shape_2[1]))
    X2 = (X2 - meanX) / stdX
    eog_train_gen_check.X2 = np.reshape(X2, (eog_train_gen_check.data_size, eog_train_gen_check.data_shape_2[0], eog_train_gen_check.data_shape_2[1]))

    X2 = eog_eval_gen.X2
    X2 = np.reshape(X2,(eog_eval_gen.data_size*eog_eval_gen.data_shape_2[0], eog_eval_gen.data_shape_2[1]))
    X2 = (X2 - meanX) / stdX
    eog_eval_gen.X2 = np.reshape(X2, (eog_eval_gen.data_size, eog_eval_gen.data_shape_2[0], eog_eval_gen.data_shape_2[1]))

    X2 = eog_test_gen.X2
    X2 = np.reshape(X2,(eog_test_gen.data_size*eog_test_gen.data_shape_2[0], eog_test_gen.data_shape_2[1]))
    X2 = (X2 - meanX) / stdX
    eog_test_gen.X2 = np.reshape(X2, (eog_test_gen.data_size, eog_test_gen.data_shape_2[0], eog_test_gen.data_shape_2[1]))

if (emg_active):
    print("emg active")
    # Initalize the EMG data generator seperately for the training, validation, and test sets
    emg_train_gen = DataGenerator(os.path.abspath(FLAGS.emg_train_data),
                                  data_shape_1=[config.deep_ntime],
                                  data_shape_2=[config.seq_frame_seq_len, config.seq_ndim],
                                  seq_len=config.epoch_seq_len, shuffle = False)
    emg_train_gen_check = DataGenerator(os.path.abspath(FLAGS.emg_train_data_check),
                                  data_shape_1=[config.deep_ntime],
                                  data_shape_2=[config.seq_frame_seq_len, config.seq_ndim],
                                  seq_len=config.epoch_seq_len, shuffle = False)
    emg_test_gen = DataGenerator(os.path.abspath(FLAGS.emg_test_data),
                                 data_shape_1=[config.deep_ntime],
                                 data_shape_2=[config.seq_frame_seq_len, config.seq_ndim],
                                 seq_len=config.epoch_seq_len, shuffle = False)
    emg_eval_gen = DataGenerator(os.path.abspath(FLAGS.emg_eval_data),
                                 data_shape_1=[config.deep_ntime],
                                 data_shape_2=[config.seq_frame_seq_len, config.seq_ndim],
                                 seq_len=config.epoch_seq_len, shuffle = False)

    emg_train_gen.X1 = np.expand_dims(emg_train_gen.X1, axis=-1) # expand feature dimension
    emg_train_gen_check.X1 = np.expand_dims(emg_train_gen_check.X1, axis=-1) # expand feature dimension
    emg_test_gen.X1 = np.expand_dims(emg_test_gen.X1, axis=-1) # expand feature dimension
    emg_eval_gen.X1 = np.expand_dims(emg_eval_gen.X1, axis=-1) # expand feature dimension

    # data normalization for time-frequency input
    X2 = emg_train_gen.X2
    X2 = np.reshape(X2,(emg_train_gen.data_size*emg_train_gen.data_shape_2[0], emg_train_gen.data_shape_2[1]))
    meanX = X2.mean(axis=0)
    stdX = X2.std(axis=0)
    X2 = (X2 - meanX) / stdX
    emg_train_gen.X2 = np.reshape(X2, (emg_train_gen.data_size, emg_train_gen.data_shape_2[0], emg_train_gen.data_shape_2[1]))

    X2 = emg_train_gen_check.X2
    X2 = np.reshape(X2,(emg_train_gen_check.data_size*emg_train_gen_check.data_shape_2[0], emg_train_gen_check.data_shape_2[1]))
    X2 = (X2 - meanX) / stdX
    emg_train_gen_check.X2 = np.reshape(X2, (emg_train_gen_check.data_size, emg_train_gen_check.data_shape_2[0], emg_train_gen_check.data_shape_2[1]))

    X2 = emg_eval_gen.X2
    X2 = np.reshape(X2,(emg_eval_gen.data_size*emg_eval_gen.data_shape_2[0], emg_eval_gen.data_shape_2[1]))
    X2 = (X2 - meanX) / stdX
    emg_eval_gen.X2 = np.reshape(X2, (emg_eval_gen.data_size, emg_eval_gen.data_shape_2[0], emg_eval_gen.data_shape_2[1]))

    X2 = emg_test_gen.X2
    X2 = np.reshape(X2,(emg_test_gen.data_size*emg_test_gen.data_shape_2[0], emg_test_gen.data_shape_2[1]))
    X2 = (X2 - meanX) / stdX
    emg_test_gen.X2 = np.reshape(X2, (emg_test_gen.data_size, emg_test_gen.data_shape_2[0], emg_test_gen.data_shape_2[1]))

# eeg always active
train_generator = eeg_train_gen
train_generator_check = eeg_train_gen_check
test_generator = eeg_test_gen
eval_generator = eeg_eval_gen

# 1-channel input case
if (not(eog_active) and not(emg_active)):
    train_generator.X1 = np.expand_dims(train_generator.X1, axis=-1) # expand channel dimension
    train_generator.data_shape_1 = train_generator.X1.shape[1:]
    train_generator_check.X1 = np.expand_dims(train_generator_check.X1, axis=-1) # expand channel dimension
    train_generator_check.data_shape_1 = train_generator_check.X1.shape[1:]
    test_generator.X1 = np.expand_dims(test_generator.X1, axis=-1) # expand channel dimension
    test_generator.data_shape_1 = test_generator.X1.shape[1:]
    eval_generator.X1 = np.expand_dims(eval_generator.X1, axis=-1) # expand channel dimension
    eval_generator.data_shape_1 = eval_generator.X1.shape[1:]
    print(train_generator.X1.shape)

    train_generator.X2 = np.expand_dims(train_generator.X2, axis=-1) # expand channel dimension
    train_generator.data_shape_2 = train_generator.X2.shape[1:]
    train_generator_check.X2 = np.expand_dims(train_generator_check.X2, axis=-1) # expand channel dimension
    train_generator_check.data_shape_2 = train_generator_check.X2.shape[1:]
    test_generator.X2 = np.expand_dims(test_generator.X2, axis=-1) # expand channel dimension
    test_generator.data_shape_2 = test_generator.X2.shape[1:]
    eval_generator.X2 = np.expand_dims(eval_generator.X2, axis=-1) # expand channel dimension
    eval_generator.data_shape_2 = eval_generator.X2.shape[1:]
    print(train_generator.X2.shape)
    nchannel = 1

# 2-channel input case
if (eog_active and not(emg_active)):
    print(train_generator.X1.shape)
    print(eog_train_gen.X1.shape)
    train_generator.X1 = np.stack((train_generator.X1, eog_train_gen.X1), axis=-1) # merge and make new dimension
    train_generator.data_shape_1 = train_generator.X1.shape[1:]
    train_generator_check.X1 = np.stack((train_generator_check.X1, eog_train_gen_check.X1), axis=-1) # merge and make new dimension
    train_generator_check.data_shape_1 = train_generator_check.X1.shape[1:]
    test_generator.X1 = np.stack((test_generator.X1, eog_test_gen.X1), axis=-1) # merge and make new dimension
    test_generator.data_shape_1 = test_generator.X1.shape[1:]
    eval_generator.X1 = np.stack((eval_generator.X1, eog_eval_gen.X1), axis=-1) # merge and make new dimension
    eval_generator.data_shape_1 = eval_generator.X1.shape[1:]
    print(train_generator.X1.shape)

    print(train_generator.X2.shape)
    print(eog_train_gen.X2.shape)
    train_generator.X2 = np.stack((train_generator.X2, eog_train_gen.X2), axis=-1) # merge and make new dimension
    train_generator.data_shape_2 = train_generator.X2.shape[1:]
    train_generator_check.X2 = np.stack((train_generator_check.X2, eog_train_gen_check.X2), axis=-1) # merge and make new dimension
    train_generator_check.data_shape_2 = train_generator_check.X2.shape[1:]
    test_generator.X2 = np.stack((test_generator.X2, eog_test_gen.X2), axis=-1) # merge and make new dimension
    test_generator.data_shape_2 = test_generator.X2.shape[1:]
    eval_generator.X2 = np.stack((eval_generator.X2, eog_eval_gen.X2), axis=-1) # merge and make new dimension
    eval_generator.data_shape_2 = eval_generator.X2.shape[1:]
    print(train_generator.X2.shape)
    nchannel = 2

# 3-channel input case
if (eog_active and emg_active):
    print(train_generator.X1.shape)
    print(eog_train_gen.X1.shape)
    print(emg_train_gen.X1.shape)
    train_generator.X1 = np.stack((train_generator.X1, eog_train_gen.X1, emg_train_gen.X1), axis=-1) # merge and make new dimension
    train_generator.data_shape_1 = train_generator.X1.shape[1:]
    train_generator_check.X1 = np.stack((train_generator_check.X1, eog_train_gen_check.X1, emg_train_gen_check.X1), axis=-1) # merge and make new dimension
    train_generator_check.data_shape_1 = train_generator_check.X1.shape[1:]
    test_generator.X1 = np.stack((test_generator.X1, eog_test_gen.X1, emg_test_gen.X1), axis=-1) # merge and make new dimension
    test_generator.data_shape_1 = test_generator.X1.shape[1:]
    eval_generator.X1 = np.stack((eval_generator.X1, eog_eval_gen.X1, emg_eval_gen.X1), axis=-1) # merge and make new dimension
    eval_generator.data_shape_1 = eval_generator.X1.shape[1:]
    print(train_generator.X1.shape)

    print(train_generator.X2.shape)
    print(eog_train_gen.X2.shape)
    print(emg_train_gen.X2.shape)
    train_generator.X2 = np.stack((train_generator.X2, eog_train_gen.X2, emg_train_gen.X2), axis=-1) # merge and make new dimension
    train_generator.data_shape_2 = train_generator.X2.shape[1:]
    train_generator_check.X2 = np.stack((train_generator_check.X2, eog_train_gen_check.X2, emg_train_gen_check.X2), axis=-1) # merge and make new dimension
    train_generator_check.data_shape_2 = train_generator_check.X2.shape[1:]
    test_generator.X2 = np.stack((test_generator.X2, eog_test_gen.X2, emg_test_gen.X2), axis=-1) # merge and make new dimension
    test_generator.data_shape_2 = test_generator.X2.shape[1:]
    eval_generator.X2 = np.stack((eval_generator.X2, eog_eval_gen.X2, emg_eval_gen.X2), axis=-1) # merge and make new dimension
    eval_generator.data_shape_2 = eval_generator.X2.shape[1:]
    print(train_generator.X2.shape)
    nchannel = 3

config.nchannel = nchannel

del eeg_train_gen
del eeg_train_gen_check
del eeg_test_gen
del eeg_eval_gen
if (eog_active):
    del eog_train_gen
    del eog_train_gen_check
    del eog_test_gen
    del eog_eval_gen
if (emg_active):
    del emg_train_gen
    del emg_train_gen_check
    del emg_test_gen
    del emg_eval_gen

# shuffle training data here
train_generator.shuffle_data()

train_batches_per_epoch = np.floor(len(train_generator.data_index) / config.batch_size).astype(np.uint32)
eval_batches_per_epoch = np.floor(len(eval_generator.data_index) / config.batch_size).astype(np.uint32)
test_batches_per_epoch = np.floor(len(test_generator.data_index) / config.batch_size).astype(np.uint32)

print("Train/Eval/Test set: {:d}/{:d}/{:d}".format(train_generator.data_size, eval_generator.data_size, test_generator.data_size))

print("Train/Eval/Test batches per epoch: {:d}/{:d}/{:d}".format(train_batches_per_epoch, eval_batches_per_epoch, test_batches_per_epoch))

# variable to keep track of performance for model selection
best_acc = 0.0

# Initialize equal weights for all three branches of the network
w1 = 1.0/3
w2 = 1.0/3
w3 = 1.0/3

# to saved the weights of the selected model
best_w1 = w1
best_w2 = w2
best_w3 = w3

with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0, allow_growth=False)
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement,
      gpu_options=gpu_options)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        net = XSleep_Net(config=config)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(config.learning_rate)
            grads_and_vars = optimizer.compute_gradients(net.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        out_dir = os.path.abspath(os.path.join(os.path.curdir,FLAGS.out_dir))
        print("Writing to {}\n".format(out_dir))

        saver = tf.train.Saver(tf.all_variables(), max_to_keep=1)

        # initialize all variables
        print("Model initialized")
        sess.run(tf.initialize_all_variables())

        def train_step(x1_batch, x2_batch, y_batch):
            """
            A single training step
            """
            print("w1 {}, w2 {}, w3 {}".format(w1, w2, w3))

            seq_frame_seq_len = np.ones(len(x1_batch)*config.epoch_seq_len,dtype=int) * config.seq_frame_seq_len
            epoch_seq_len = np.ones(len(x1_batch),dtype=int) * config.epoch_seq_len
            feed_dict = {
              net.input_x1: x1_batch,
              net.input_x2: x2_batch,
              net.input_y: y_batch,
              net.seq_dropout_keep_prob_rnn: config.seq_dropout_keep_prob_rnn,
              net.epoch_seq_len: epoch_seq_len,
              net.seq_frame_seq_len: seq_frame_seq_len,
              net.deep_dropout: config.deep_dropout,
              net.w1 : w1,
              net.w2 : w2,
              net.w3 : w3,
              net.istraining: 1
            }
            _, step, output_loss, total_loss, accuracy = sess.run(
               [train_op, global_step, net.output_loss, net.loss, net.accuracy],
               feed_dict)
            return step, output_loss, total_loss, accuracy

        def dev_step(x1_batch, x2_batch, y_batch):
            """
            A single testing step
            """
            seq_frame_seq_len = np.ones(len(x1_batch)*config.epoch_seq_len,dtype=int) * config.seq_frame_seq_len
            epoch_seq_len = np.ones(len(x1_batch),dtype=int) * config.epoch_seq_len
            feed_dict = {
                net.input_x1: x1_batch,
                net.input_x2: x2_batch,
                net.input_y: y_batch,
                net.seq_dropout_keep_prob_rnn: 1.0,
                net.epoch_seq_len: epoch_seq_len,
                net.seq_frame_seq_len: seq_frame_seq_len,
                net.deep_dropout: config.deep_dropout,
                net.w1 : w1,
                net.w2 : w2,
                net.w3 : w3,
                net.istraining: 0
            }
            output_loss1, output_loss2, output_loss3, output_loss, total_loss, \
            deep_yhat, seq_yhat, joint_yhat, yhat = sess.run(
                   [net.output_loss1, net.output_loss2, net.output_loss3, net.output_loss, net.loss,
                    net.deep_predictions, net.seq_predictions, net.joint_predictions, net.predictions], feed_dict)
            return output_loss1, output_loss2, output_loss3, output_loss, total_loss, deep_yhat, seq_yhat, joint_yhat, yhat

        def evaluate(gen, log_filename):
            # Validate the model on the entire data of DataGenerator gen
            output_loss1 =0
            output_loss2 =0
            output_loss3 =0
            output_loss =0
            total_loss = 0
            deep_yhat = np.zeros([config.epoch_seq_len, len(gen.data_index)])
            seq_yhat = np.zeros([config.epoch_seq_len, len(gen.data_index)])
            joint_yhat = np.zeros([config.epoch_seq_len, len(gen.data_index)])
            yhat = np.zeros([config.epoch_seq_len, len(gen.data_index)])

            num_batch_per_epoch = np.floor(len(gen.data_index) / (10*config.batch_size)).astype(np.uint32)
            test_step = 1
            while test_step < num_batch_per_epoch:
                x1_batch, x2_batch, y_batch, label_batch_ = gen.next_batch(10*config.batch_size)
                output_loss1_, output_loss2_, output_loss3_, output_loss_, total_loss_, \
                deep_yhat_, seq_yhat_, joint_yhat_, yhat_ = dev_step(x1_batch, x2_batch, y_batch)
                output_loss1 += output_loss1_
                output_loss2 += output_loss2_
                output_loss3 += output_loss3_
                output_loss += output_loss_
                total_loss += total_loss_
                for n in range(config.epoch_seq_len):
                    deep_yhat[n, (test_step-1)*10*config.batch_size : test_step*10*config.batch_size] = deep_yhat_[n]
                    seq_yhat[n, (test_step-1)*10*config.batch_size : test_step*10*config.batch_size] = seq_yhat_[n]
                    joint_yhat[n, (test_step-1)*10*config.batch_size : test_step*10*config.batch_size] = joint_yhat_[n]
                    yhat[n, (test_step-1)*10*config.batch_size : test_step*10*config.batch_size] = yhat_[n]
                test_step += 1
            if(gen.pointer < len(gen.data_index)):
                actual_len, x1_batch, x2_batch, y_batch, label_batch_ = gen.rest_batch(config.batch_size)
                output_loss1_, output_loss2_, output_loss3_, output_loss_, total_loss_, \
                deep_yhat_, seq_yhat_, joint_yhat_, yhat_ = dev_step(x1_batch, x2_batch, y_batch)
                output_loss1 += output_loss1_
                output_loss2 += output_loss2_
                output_loss3 += output_loss3_
                output_loss += output_loss_
                total_loss += total_loss_
                for n in range(config.epoch_seq_len):
                    deep_yhat[n, (test_step-1)*10*config.batch_size : len(gen.data_index)] = deep_yhat_[n]
                    seq_yhat[n, (test_step-1)*10*config.batch_size : len(gen.data_index)] = seq_yhat_[n]
                    joint_yhat[n, (test_step-1)*10*config.batch_size : len(gen.data_index)] = joint_yhat_[n]
                    yhat[n, (test_step-1)*10*config.batch_size : len(gen.data_index)] = yhat_[n]
            deep_yhat = deep_yhat + 1
            seq_yhat = seq_yhat + 1
            joint_yhat = joint_yhat + 1
            yhat = yhat + 1
            acc1 = 0
            acc2 = 0
            acc3 = 0
            acc = 0
            with open(os.path.join(out_dir, log_filename), "a") as text_file:
                text_file.write("{:g} {:g} {:g} {:g} {:g} ".format(output_loss1, output_loss2, output_loss3, output_loss, total_loss))
                for n in range(config.epoch_seq_len):
                    acc_n = accuracy_score(deep_yhat[n,:], gen.label[gen.data_index - (config.epoch_seq_len - 1) + n]) # due to zero-indexing
                    text_file.write("{:g} ".format(acc_n))
                    acc1 += acc_n
                for n in range(config.epoch_seq_len):
                    acc_n = accuracy_score(seq_yhat[n,:], gen.label[gen.data_index - (config.epoch_seq_len - 1) + n]) # due to zero-indexing
                    text_file.write("{:g} ".format(acc_n))
                    acc2 += acc_n
                for n in range(config.epoch_seq_len):
                    acc_n = accuracy_score(joint_yhat[n,:], gen.label[gen.data_index - (config.epoch_seq_len - 1) + n]) # due to zero-indexing
                    text_file.write("{:g} ".format(acc_n))
                    acc3 += acc_n
                for n in range(config.epoch_seq_len):
                    acc_n = accuracy_score(yhat[n,:], gen.label[gen.data_index - (config.epoch_seq_len - 1) + n]) # due to zero-indexing
                    if n == config.epoch_seq_len - 1:
                        text_file.write("{:g} \n".format(acc_n))
                    else:
                        text_file.write("{:g} ".format(acc_n))
                    acc += acc_n
            acc1 /= config.epoch_seq_len
            acc2 /= config.epoch_seq_len
            acc3 /= config.epoch_seq_len
            acc /= config.epoch_seq_len
            return acc1, acc2, acc3, acc, deep_yhat, seq_yhat, joint_yhat, yhat, output_loss1, output_loss2, output_loss3, output_loss, total_loss

        # Initial evaluation (without any network training)
        print("{} Start validation 0".format(datetime.now()))
        # on the evaluation set
        eval_acc1_0, eval_acc2_0, eval_acc3_0, eval_acc_0, \
        eval_yhat1_0, eval_yhat2_0, eval_yhat3_0, eval_yhat_0, \
        eval_output_loss1_0, eval_output_loss2_0, eval_output_loss3_0, eval_output_loss_0, eval_total_loss_0 = \
            evaluate(gen=eval_generator, log_filename="eval_result_log.txt")
        # on the training checking set
        train_acc1_0, train_acc2_0, train_acc3_0, train_acc_0, \
        train_yhat1_0, train_yhat2_0, train_yhat3_0, train_yhat_0, \
        train_output_loss1_0, train_output_loss2_0, train_output_loss3_0, train_output_loss_0, train_total_loss_0 = \
            evaluate(gen=train_generator_check, log_filename="train_result_log.txt")
        # on the test set
        test_acc1_0, test_acc2_0, test_acc3_0, test_acc_0, \
        test_yhat1_0, test_yhat2_0, test_yhat3_0, test_yhat_0, \
        test_output_loss1_0, test_output_loss2_0, test_output_loss3_0, test_output_loss_0, test_total_loss_0 = \
            evaluate(gen=test_generator, log_filename="test_result_log.txt")
        train_generator_check.reset_pointer()
        eval_generator.reset_pointer()
        test_generator.reset_pointer()


        start_time = time.time()
        # Loop over number of epochs
        for epoch in range(config.training_epoch):
            print("{} Epoch number: {}".format(datetime.now(), epoch + 1))
            step = 1
            while step < train_batches_per_epoch:
                # Get a batch and train the network
                x1_batch, x2_batch, y_batch, label_batch = train_generator.next_batch(config.batch_size)
                train_step_, train_output_loss_, train_total_loss_, train_acc_ = train_step(x1_batch, x2_batch, y_batch)
                time_str = datetime.now().isoformat()

                # Accuracy average over sequence length
                acc_ = 0
                for n in range(config.epoch_seq_len):
                    acc_ += train_acc_[n]
                acc_ /= config.epoch_seq_len

                print("{}: step {}, output_loss {}, total_loss {} acc {}".format(time_str, train_step_, train_output_loss_, train_total_loss_, acc_))
                step += 1

                current_step = tf.train.global_step(sess, global_step)
                if current_step % config.evaluate_every == 0:
                    # Validate the current model
                    print("{} Start validation".format(datetime.now()))

                    # on the evaluation set
                    eval_acc1, eval_acc2, eval_acc3, eval_acc, \
                    eval_yhat1, eval_yhat2, eval_yhat3, eval_yhat, \
                    eval_output_loss1, eval_output_loss2, eval_output_loss3, eval_output_loss, eval_total_loss = \
                        evaluate(gen=eval_generator, log_filename="eval_result_log.txt")
                    # on the training checking set
                    train_acc1, train_acc2, train_acc3, train_acc, \
                    train_yhat1, train_yhat2, train_yhat3, train_yhat, \
                    train_output_loss1, train_output_loss2, train_output_loss3, train_output_loss, train_total_loss = \
                        evaluate(gen=train_generator_check, log_filename="train_result_log.txt")
                    # on the test set
                    test_acc1, test_acc2, test_acc3, test_acc, \
                    test_yhat1, test_yhat2, test_yhat3, test_yhat, \
                    test_output_loss1, test_output_loss2, test_output_loss3, test_output_loss, test_total_loss = \
                        evaluate(gen=test_generator, log_filename="test_result_log.txt")

                    if(eval_acc >= best_acc):
                        best_acc = eval_acc
                        best_w1 = w1
                        best_w2 = w2
                        best_w3 = w3
                        checkpoint_name = os.path.join(checkpoint_path, 'model_step' + str(current_step) +'.ckpt')
                        save_path = saver.save(sess, checkpoint_name)

                        print("Best model updated")
                        source_file = checkpoint_name
                        dest_file = os.path.join(checkpoint_path, 'best_model_acc')
                        shutil.copy(source_file + '.data-00000-of-00001', dest_file + '.data-00000-of-00001')
                        shutil.copy(source_file + '.index', dest_file + '.index')
                        shutil.copy(source_file + '.meta', dest_file + '.meta')

                    # compute the ratio of overfitting and convergence of each network branch and update the weights
                    o1 = (train_output_loss1_0 - train_output_loss1)/len(train_generator_check.data_index) - (eval_output_loss1_0 - eval_output_loss1)/len(eval_generator.data_index)
                    g1 = (eval_output_loss1_0 - eval_output_loss1)/len(eval_generator.data_index)
                    w1_ = abs(g1/(o1*o1 + 1e-6)) # to avoid devided by 0

                    o2 = (train_output_loss2_0 - train_output_loss2)/len(train_generator_check.data_index) - (eval_output_loss2_0 - eval_output_loss2)/len(eval_generator.data_index)
                    g2 = (eval_output_loss2_0 - eval_output_loss2)/len(eval_generator.data_index)
                    w2_ = abs(g2/(o2*o2 + 1e-6)) # to avoid devided by 0

                    o3 = (train_output_loss3_0 - train_output_loss3)/len(train_generator_check.data_index) - (eval_output_loss3_0 - eval_output_loss3)/len(eval_generator.data_index)
                    g3 = (eval_output_loss3_0 - eval_output_loss3)/len(eval_generator.data_index)
                    w3_ = abs(g3/(o3*o3 + 1e-6)) # to avoid devided by 0

                    # weight normalization
                    w1 = w1_/(w1_ + w2_ + w3_)
                    w2 = w2_/(w1_ + w2_ + w3_)
                    w3 = w3_/(w1_ + w2_ + w3_)

                    with open(os.path.join(out_dir, "w.txt"), "a") as text_file:
                        text_file.write("{:g} {:g} {:g} {:g} {:g} {:g} {:g} {:g} {:g} {:g} {:g} {:g} \n".format(o1, g1, w1_, w1, o2, g2, w2_, w2, o3, g3, w3_, w3))

                    test_generator.reset_pointer()
                    eval_generator.reset_pointer()
                    train_generator_check.reset_pointer()
            train_generator.reset_pointer()

        end_time = time.time()
        with open(os.path.join(out_dir, "training_time.txt"), "a") as text_file:
            text_file.write("{:g}\n".format((end_time - start_time)))
        with open(os.path.join(out_dir, "final_w.txt"), "a") as text_file:
            text_file.write("{:g} {:g} {:g}\n".format(best_w1, best_w2, best_w3))
