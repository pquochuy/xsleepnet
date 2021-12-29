import os
import numpy as np
import tensorflow as tf

import shutil, sys
from datetime import datetime
import h5py

from xsleepnet import XSleepNet
from xsleepnet_config import Config

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score

from datagenerator_from_list_v2 import DataGenerator

from scipy.io import loadmat, savemat


# Parameters
# ==================================================

# Misc Parameters
tf.app.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.app.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# My Parameters
tf.app.flags.DEFINE_string("eeg_train_data", "../train_data.mat", "Point to directory of input data")
tf.app.flags.DEFINE_string("eeg_test_data", "../test_data.mat", "Point to directory of input data")
tf.app.flags.DEFINE_string("eog_train_data", "../train_data.mat", "Point to directory of input data")
tf.app.flags.DEFINE_string("eog_test_data", "../test_data.mat", "Point to directory of input data")
tf.app.flags.DEFINE_string("emg_train_data", "../train_data.mat", "Point to directory of input data")
tf.app.flags.DEFINE_string("emg_test_data", "../test_data.mat", "Point to directory of input data")
tf.app.flags.DEFINE_string("out_dir", "./output/", "Point to output directory")
tf.app.flags.DEFINE_string("checkpoint_dir", "./checkpoint/", "Point to checkpoint directory")

# seqsleepnet settings
tf.app.flags.DEFINE_float("dropout_rnn", 0.75, "Dropout keep probability (default: 0.75)")
tf.app.flags.DEFINE_integer("seq_nfilter", 32, "Sequence length (default: 20)")
tf.app.flags.DEFINE_integer("seq_nhidden1", 64, "Sequence length (default: 20)")
tf.app.flags.DEFINE_integer("seq_attention_size1", 32, "Sequence length (default: 20)")
tf.app.flags.DEFINE_integer("seq_nhidden2", 64, "Sequence length (default: 20)")

# fcnnrnn settings
tf.app.flags.DEFINE_float("dropout_cnn", 0.5, "Dropout keep probability (default: 0.75)")
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
config.dropout_rnn = FLAGS.dropout_rnn
config.epoch_seq_len = FLAGS.seq_len
config.seq_nfilter = FLAGS.seq_nfilter
config.seq_nhidden1 = FLAGS.seq_nhidden1
config.seq_nhidden2 = FLAGS.seq_nhidden2
config.seq_attention_size1 = FLAGS.seq_attention_size1

config.dropout_cnn = FLAGS.dropout_cnn
config.deep_nhidden = FLAGS.deep_nhidden

eeg_active = ((FLAGS.eeg_train_data != "") and (FLAGS.eeg_test_data != ""))
eog_active = ((FLAGS.eog_train_data != "") and (FLAGS.eog_test_data != ""))
emg_active = ((FLAGS.emg_train_data != "") and (FLAGS.emg_test_data != ""))

if (eeg_active):
    print("eeg active")
    # Initalize the data generator seperately for the training, validation, and test sets
    eeg_train_gen = DataGenerator(os.path.abspath(FLAGS.eeg_train_data),
                                  data_shape_1=[config.deep_ntime],
                                  data_shape_2=[config.seq_frame_seq_len, config.seq_ndim],
                                  seq_len=config.epoch_seq_len, shuffle = False)
    eeg_test_gen = DataGenerator(os.path.abspath(FLAGS.eeg_test_data),
                                 data_shape_1=[config.deep_ntime],
                                 data_shape_2=[config.seq_frame_seq_len, config.seq_ndim],
                                 seq_len=config.epoch_seq_len, shuffle = False)
    
    # data normalization for time-frequency here
    X2 = eeg_train_gen.X2
    X2 = np.reshape(X2,(eeg_train_gen.data_size*eeg_train_gen.data_shape_2[0], eeg_train_gen.data_shape_2[1]))
    meanX = X2.mean(axis=0)
    stdX = X2.std(axis=0)
    X2 = (X2 - meanX) / stdX
    eeg_train_gen.X2 = np.reshape(X2, (eeg_train_gen.data_size, eeg_train_gen.data_shape_2[0], eeg_train_gen.data_shape_2[1]))

    X2 = eeg_test_gen.X2
    X2 = np.reshape(X2,(eeg_test_gen.data_size*eeg_test_gen.data_shape_2[0], eeg_test_gen.data_shape_2[1]))
    X2 = (X2 - meanX) / stdX
    eeg_test_gen.X2 = np.reshape(X2, (eeg_test_gen.data_size, eeg_test_gen.data_shape_2[0], eeg_test_gen.data_shape_2[1]))

if (eog_active):
    print("eog active")
    # Initalize the data generator seperately for the training, validation, and test sets
    eog_train_gen = DataGenerator(os.path.abspath(FLAGS.eog_train_data),
                                  data_shape_1=[config.deep_ntime],
                                  data_shape_2=[config.seq_frame_seq_len, config.seq_ndim],
                                  seq_len=config.epoch_seq_len, shuffle = False)
    eog_test_gen = DataGenerator(os.path.abspath(FLAGS.eog_test_data),
                                 data_shape_1=[config.deep_ntime],
                                 data_shape_2=[config.seq_frame_seq_len, config.seq_ndim],
                                 seq_len=config.epoch_seq_len, shuffle = False)

    # data normalization for time-frequency here
    X2 = eog_train_gen.X2
    X2 = np.reshape(X2,(eog_train_gen.data_size*eog_train_gen.data_shape_2[0], eog_train_gen.data_shape_2[1]))
    meanX = X2.mean(axis=0)
    stdX = X2.std(axis=0)
    X2 = (X2 - meanX) / stdX
    eog_train_gen.X2 = np.reshape(X2, (eog_train_gen.data_size, eog_train_gen.data_shape_2[0], eog_train_gen.data_shape_2[1]))

    X2 = eog_test_gen.X2
    X2 = np.reshape(X2,(eog_test_gen.data_size*eog_test_gen.data_shape_2[0], eog_test_gen.data_shape_2[1]))
    X2 = (X2 - meanX) / stdX
    eog_test_gen.X2 = np.reshape(X2, (eog_test_gen.data_size, eog_test_gen.data_shape_2[0], eog_test_gen.data_shape_2[1]))

if (emg_active):
    print("emg active")
    # Initalize the data generator seperately for the training, validation, and test sets
    emg_train_gen = DataGenerator(os.path.abspath(FLAGS.emg_train_data),
                                  data_shape_1=[config.deep_ntime],
                                  data_shape_2=[config.seq_frame_seq_len, config.seq_ndim],
                                  seq_len=config.epoch_seq_len, shuffle = False)
    emg_test_gen = DataGenerator(os.path.abspath(FLAGS.emg_test_data),
                                 data_shape_1=[config.deep_ntime],
                                 data_shape_2=[config.seq_frame_seq_len, config.seq_ndim],
                                 seq_len=config.epoch_seq_len, shuffle = False)

    # data normalization here
    X2 = emg_train_gen.X2
    X2 = np.reshape(X2,(emg_train_gen.data_size*emg_train_gen.data_shape_2[0], emg_train_gen.data_shape_2[1]))
    meanX = X2.mean(axis=0)
    stdX = X2.std(axis=0)
    X2 = (X2 - meanX) / stdX
    emg_train_gen.X2 = np.reshape(X2, (emg_train_gen.data_size, emg_train_gen.data_shape_2[0], emg_train_gen.data_shape_2[1]))

    X2 = emg_test_gen.X2
    X2 = np.reshape(X2,(emg_test_gen.data_size*emg_test_gen.data_shape_2[0], emg_test_gen.data_shape_2[1]))
    X2 = (X2 - meanX) / stdX
    emg_test_gen.X2 = np.reshape(X2, (emg_test_gen.data_size, emg_test_gen.data_shape_2[0], emg_test_gen.data_shape_2[1]))

# eeg always active
train_generator = eeg_train_gen
test_generator = eeg_test_gen

if (not(eog_active) and not(emg_active)):
    train_generator.X1 = np.expand_dims(train_generator.X1, axis=-1) # expand channel dimension
    train_generator.data_shape_1 = train_generator.X1.shape[1:]
    test_generator.X1 = np.expand_dims(test_generator.X1, axis=-1) # expand channel dimension
    test_generator.data_shape_1 = test_generator.X1.shape[1:]
    print(train_generator.X1.shape)

    train_generator.X2 = np.expand_dims(train_generator.X2, axis=-1) # expand channel dimension
    train_generator.data_shape_2 = train_generator.X2.shape[1:]
    test_generator.X2 = np.expand_dims(test_generator.X2, axis=-1) # expand channel dimension
    test_generator.data_shape_2 = test_generator.X2.shape[1:]
    print(train_generator.X2.shape)
    nchannel = 1

if (eog_active and not(emg_active)):
    print(train_generator.X1.shape)
    print(eog_train_gen.X1.shape)
    train_generator.X1 = np.stack((train_generator.X1, eog_train_gen.X1), axis=-1) # merge and make new dimension
    train_generator.data_shape_1 = train_generator.X1.shape[1:]
    test_generator.X1 = np.stack((test_generator.X1, eog_test_gen.X1), axis=-1) # merge and make new dimension
    test_generator.data_shape_1 = test_generator.X1.shape[1:]
    print(train_generator.X1.shape)

    print(train_generator.X2.shape)
    print(eog_train_gen.X2.shape)
    train_generator.X2 = np.stack((train_generator.X2, eog_train_gen.X2), axis=-1) # merge and make new dimension
    train_generator.data_shape_2 = train_generator.X2.shape[1:]
    test_generator.X2 = np.stack((test_generator.X2, eog_test_gen.X2), axis=-1) # merge and make new dimension
    test_generator.data_shape_2 = test_generator.X2.shape[1:]
    print(train_generator.X2.shape)
    nchannel = 2

if (eog_active and emg_active):
    print(train_generator.X1.shape)
    print(eog_train_gen.X1.shape)
    print(emg_train_gen.X1.shape)
    train_generator.X1 = np.stack((train_generator.X1, eog_train_gen.X1, emg_train_gen.X1), axis=-1) # merge and make new dimension
    train_generator.data_shape_1 = train_generator.X1.shape[1:]
    test_generator.X1 = np.stack((test_generator.X1, eog_test_gen.X1, emg_test_gen.X1), axis=-1) # merge and make new dimension
    test_generator.data_shape_1 = test_generator.X1.shape[1:]
    print(train_generator.X1.shape)

    print(train_generator.X2.shape)
    print(eog_train_gen.X2.shape)
    print(emg_train_gen.X2.shape)
    train_generator.X2 = np.stack((train_generator.X2, eog_train_gen.X2, emg_train_gen.X2), axis=-1) # merge and make new dimension
    train_generator.data_shape_2 = train_generator.X2.shape[1:]
    test_generator.X2 = np.stack((test_generator.X2, eog_test_gen.X2, emg_test_gen.X2), axis=-1) # merge and make new dimension
    test_generator.data_shape_2 = test_generator.X2.shape[1:]
    print(train_generator.X2.shape)
    nchannel = 3

config.nchannel = nchannel

del eeg_train_gen
del eeg_test_gen
if (eog_active):
    del eog_train_gen
    del eog_test_gen
if (emg_active):
    del emg_train_gen
    del emg_test_gen

# shuffle training data here
del train_generator
test_batches_per_epoch = np.floor(len(test_generator.data_index) / config.batch_size).astype(np.uint32)
print("Test set: {:d}".format(test_generator.data_size))
print("/Test batches per epoch: {:d}".format(test_batches_per_epoch))


with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0, allow_growth=False)
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement,
      gpu_options=gpu_options)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        net = XSleepNet(config=config)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(config.learning_rate)
            grads_and_vars = optimizer.compute_gradients(net.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
            
        out_dir = os.path.abspath(os.path.join(os.path.curdir,FLAGS.out_dir))
        print("Writing to {}\n".format(out_dir))


        def dev_step(x1_batch, x2_batch, y_batch):
            seq_frame_seq_len = np.ones(len(x1_batch)*config.epoch_seq_len,dtype=int) * config.seq_frame_seq_len
            epoch_seq_len = np.ones(len(x1_batch),dtype=int) * config.epoch_seq_len
            feed_dict = {
                net.input_x1: x1_batch,
                net.input_x2: x2_batch,
                net.input_y: y_batch,
                net.dropout_rnn: 1.0,
                net.epoch_seq_len: epoch_seq_len,
                net.seq_frame_seq_len: seq_frame_seq_len,
                net.dropout_cnn: 1.0,
                net.w1 : 1./3,
                net.w2 : 1./3,
                net.w3 : 1./3,
                net.istraining: 0
            }
            output_loss1, output_loss2, output_loss3, output_loss, total_loss, \
            deep_yhat, seq_yhat, joint_yhat, yhat, deep_score, seq_score, joint_score, score = sess.run(
                   [net.deep_loss, net.seq_loss, net.joint_loss, net.output_loss, net.loss,
                    net.deep_predictions, net.seq_predictions, net.joint_predictions, net.predictions,
                    net.deep_scores, net.seq_scores, net.joint_scores, net.score], feed_dict)
            return output_loss1, output_loss2, output_loss3, output_loss, total_loss, \
                   deep_yhat, seq_yhat, joint_yhat, yhat, \
                   deep_score, seq_score, joint_score, score

        def evaluate(gen):
            # Validate the model on the entire evaluation test set after each epoch
            output_loss1 =0
            output_loss2 =0
            output_loss3 =0
            output_loss =0
            total_loss = 0
            deep_yhat = np.zeros([config.epoch_seq_len, len(gen.data_index)])
            seq_yhat = np.zeros([config.epoch_seq_len, len(gen.data_index)])
            joint_yhat = np.zeros([config.epoch_seq_len, len(gen.data_index)])
            yhat = np.zeros([config.epoch_seq_len, len(gen.data_index)])

            deep_score = np.zeros([config.epoch_seq_len, len(test_generator.data_index), config.nclass])
            seq_score = np.zeros([config.epoch_seq_len, len(test_generator.data_index), config.nclass])
            joint_score = np.zeros([config.epoch_seq_len, len(test_generator.data_index), config.nclass])
            score = np.zeros([config.epoch_seq_len, len(test_generator.data_index), config.nclass])

            factor = 10
            
            num_batch_per_epoch = np.floor(len(gen.data_index) / (factor*config.batch_size)).astype(np.uint32)
            test_step = 1
            while test_step < num_batch_per_epoch:
                x1_batch, x2_batch, y_batch, label_batch_ = gen.next_batch(factor*config.batch_size)
                output_loss1_, output_loss2_, output_loss3_, output_loss_, total_loss_, \
                deep_yhat_, seq_yhat_, joint_yhat_, yhat_, \
                deep_score_, seq_score_, joint_score_, score_ = dev_step(x1_batch, x2_batch, y_batch)
                output_loss1 += output_loss1_
                output_loss2 += output_loss2_
                output_loss3 += output_loss3_
                output_loss += output_loss_
                total_loss += total_loss_
                for n in range(config.epoch_seq_len):
                    deep_yhat[n, (test_step-1)*factor*config.batch_size : test_step*factor*config.batch_size] = deep_yhat_[n]
                    seq_yhat[n, (test_step-1)*factor*config.batch_size : test_step*factor*config.batch_size] = seq_yhat_[n]
                    joint_yhat[n, (test_step-1)*factor*config.batch_size : test_step*factor*config.batch_size] = joint_yhat_[n]
                    yhat[n, (test_step-1)*factor*config.batch_size : test_step*factor*config.batch_size] = yhat_[n]

                    deep_score[n, (test_step-1)*factor*config.batch_size : test_step*factor*config.batch_size,:] = deep_score_[n]
                    seq_score[n, (test_step-1)*factor*config.batch_size : test_step*factor*config.batch_size,:] = seq_score_[n]
                    joint_score[n, (test_step-1)*factor*config.batch_size : test_step*factor*config.batch_size,:] = joint_score_[n]
                    score[n, (test_step-1)*factor*config.batch_size : test_step*factor*config.batch_size,:] = score_[n]
                test_step += 1
            if(gen.pointer < len(gen.data_index)):
                actual_len, x1_batch, x2_batch, y_batch, label_batch_ = gen.rest_batch(config.batch_size)
                output_loss1_, output_loss2_, output_loss3_, output_loss_, total_loss_, \
                deep_yhat_, seq_yhat_, joint_yhat_, yhat_, \
                deep_score_, seq_score_, joint_score_, score_ = dev_step(x1_batch, x2_batch, y_batch)
                output_loss1 += output_loss1_
                output_loss2 += output_loss2_
                output_loss3 += output_loss3_
                output_loss += output_loss_
                total_loss += total_loss_
                for n in range(config.epoch_seq_len):
                    deep_yhat[n, (test_step-1)*factor*config.batch_size : len(gen.data_index)] = deep_yhat_[n]
                    seq_yhat[n, (test_step-1)*factor*config.batch_size : len(gen.data_index)] = seq_yhat_[n]
                    joint_yhat[n, (test_step-1)*factor*config.batch_size : len(gen.data_index)] = joint_yhat_[n]
                    yhat[n, (test_step-1)*factor*config.batch_size : len(gen.data_index)] = yhat_[n]

                    deep_score[n, (test_step-1)*factor*config.batch_size : len(gen.data_index),:] = deep_score_[n]
                    seq_score[n, (test_step-1)*factor*config.batch_size : len(gen.data_index),:] = seq_score_[n]
                    joint_score[n, (test_step-1)*factor*config.batch_size : len(gen.data_index),:] = joint_score_[n]
                    score[n, (test_step-1)*factor*config.batch_size : len(gen.data_index),:] = score_[n]
            deep_yhat = deep_yhat + 1
            seq_yhat = seq_yhat + 1
            joint_yhat = joint_yhat + 1
            yhat = yhat + 1
            acc1 = 0
            acc2 = 0
            acc3 = 0
            acc = 0
            for n in range(config.epoch_seq_len):
                acc_n = accuracy_score(deep_yhat[n,:], gen.label[gen.data_index - (config.epoch_seq_len - 1) + n]) # due to zero-indexing
                acc1 += acc_n
            for n in range(config.epoch_seq_len):
                acc_n = accuracy_score(seq_yhat[n,:], gen.label[gen.data_index - (config.epoch_seq_len - 1) + n]) # due to zero-indexing
                acc2 += acc_n
            for n in range(config.epoch_seq_len):
                acc_n = accuracy_score(joint_yhat[n,:], gen.label[gen.data_index - (config.epoch_seq_len - 1) + n]) # due to zero-indexing
                acc3 += acc_n
            for n in range(config.epoch_seq_len):
                acc_n = accuracy_score(yhat[n,:], gen.label[gen.data_index - (config.epoch_seq_len - 1) + n]) # due to zero-indexing
                acc += acc_n
            acc1 /= config.epoch_seq_len
            acc2 /= config.epoch_seq_len
            acc3 /= config.epoch_seq_len
            acc /= config.epoch_seq_len
            return acc1, acc2, acc3, acc, \
                   deep_yhat, seq_yhat, joint_yhat, yhat, \
                   deep_score, seq_score, joint_score, score, \
                   output_loss1, output_loss2, output_loss3, output_loss, total_loss

        saver = tf.train.Saver(tf.all_variables())
        # Load saved model to continue training or initialize all variables
        best_dir = os.path.join(checkpoint_path, "best_model")
        saver.restore(sess, best_dir)
        print("Model joint loaded")

        deep_acc, seq_acc, joint_acc, test_acc, \
        deep_yhat, seq_yhat, joint_yhat, test_yhat,\
        deep_score, seq_score, joint_score, test_score,\
        deep_loss, seq_loss, joint_loss, test_loss, test_total_loss = \
            evaluate(gen=test_generator)

        savemat(os.path.join(out_path, "test_ret_joint.mat"), dict(yhat = test_yhat, acc = test_acc, score = test_score, output_loss = test_loss,
                                                             deep_yhat = deep_yhat, deep_acc = deep_acc, deep_score = deep_score, deep_loss = deep_loss,
                                                             seq_yhat = seq_yhat, seq_acc = seq_acc, seq_score = seq_score, seq_loss = seq_loss,
                                                             joint_yhat = joint_yhat, joint_acc = joint_acc, joint_score = joint_score, joint_loss = joint_loss,
                                                             total_loss = test_total_loss))
        test_generator.reset_pointer()

