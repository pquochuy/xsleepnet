import numpy as np
import warnings

class GradientPolicy2(object):
    def __init__(self, train_size=0, valid_size=0, average_win=20, hist_size=20):
        self.train_loss = np.array([])
        self.valid_loss = np.array([])
        self.smoothed_train_loss = np.array([])
        self.smoothed_valid_loss = np.array([])
        self.train_size = train_size
        self.valid_size = valid_size
        self.average_win = average_win
        self.hist_size = hist_size
        self.train_slope_ref = None
        self.valid_slope_ref = None

    def add_point(self, train_loss_val, valid_loss_val):
        self.train_loss = np.append(self.train_loss, [train_loss_val])
        self.valid_loss = np.append(self.valid_loss, [valid_loss_val])
        self._moving_average_smoothing()

    def _moving_average_smoothing(self):
        size = len(self.train_loss)
        smoothed_val = np.mean(self.train_loss[np.max([0, (size - self.average_win)]):])
        self.smoothed_train_loss = np.append(self.smoothed_train_loss, [smoothed_val])
        size = len(self.valid_loss)
        smoothed_val = np.mean(self.valid_loss[np.max([0, (size - self.average_win)]):])
        self.smoothed_valid_loss = np.append(self.smoothed_valid_loss, [smoothed_val])

    def _line_fit(self, train_loss, valid_loss):
        size = len(train_loss)
        train_val = train_loss[np.max([0, (size - self.hist_size)]):]
        t = np.arange(np.min([len(train_val), self.hist_size]))
        p_train = np.polyfit(t, train_val, 1)
        assert (len(p_train) == 2)
        p_train = p_train[0]

        size = len(valid_loss)
        valid_val = valid_loss[np.max([0, (size - self.hist_size)]):]
        t = np.arange(np.min([len(valid_val), self.hist_size]))
        p_valid = np.polyfit(t, valid_val, 1)
        assert (len(p_valid) == 2)
        p_valid = p_valid[0]

        return p_train, p_valid

    def compute_weight(self):
        N = len(self.smoothed_train_loss)
        if (self.train_slope_ref is None and self.valid_slope_ref is None):
            train_loss = self.smoothed_train_loss[max([0, N - self.hist_size - 1]): -1]
            valid_loss = self.smoothed_valid_loss[max([0, N - self.hist_size - 1]): -1]
            self.train_slope_ref, self.valid_slope_ref = self._line_fit(train_loss, valid_loss)

        cur_train_loss = self.smoothed_train_loss[max([0, N - self.hist_size]):]
        cur_valid_loss = self.smoothed_valid_loss[max([0, N - self.hist_size]):]
        train_slope, valid_slope = self._line_fit(cur_train_loss, cur_valid_loss)

        Ok = (valid_slope - train_slope) - (self.valid_slope_ref - self.train_slope_ref)
        Gk = valid_slope - self.valid_slope_ref

        w = Gk / (Ok * Ok + 1e-6)
        if (w < 0.):
            w = 0.

        # update references
        if (self.valid_slope_ref > valid_slope):
            self.train_slope_ref = train_slope
            self.valid_slope_ref = valid_slope

        return w, Gk, Ok
