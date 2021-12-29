import numpy as np
import warnings

class GradientPolicy1(object):
    def __init__(self, train_size=0, valid_size=0):
        self.train_loss = np.array([])
        self.valid_loss = np.array([])
        self.train_size = train_size
        self.valid_size = valid_size

    def add_point(self, train_loss_val, valid_loss_val):
        self.train_loss = np.append(self.train_loss, [train_loss_val])
        self.valid_loss = np.append(self.valid_loss, [valid_loss_val])

    def compute_weight(self):
        # compute
        Ok = (self.train_loss[0] - self.train_loss[-1]) / self.train_size - \
             (self.valid_loss[0] - self.valid_loss[-1]) / self.valid_size
        Gk = (self.valid_loss[0] - self.valid_loss[-1]) / self.valid_size
        w = Gk / (Ok * Ok + 1e-6) # to avoid devided by 0
        if (w < 0.):
            w = 0.
        return w, Gk, Ok
