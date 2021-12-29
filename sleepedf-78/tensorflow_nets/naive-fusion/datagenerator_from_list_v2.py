import numpy as np
#from scipy.io import loadmat
import h5py

class DataGenerator:
    def __init__(self, filelist, data_shape_1=np.array([3000]), data_shape_2=np.array([29, 129]), seq_len = 20, shuffle=False, test_mode=False):

        # Init params
        self.shuffle = shuffle
        self.filelist = filelist
        self.data_shape_1 = data_shape_1
        self.data_shape_2 = data_shape_2
        self.pointer = 0
        self.X1 = np.array([])
        self.X2 = np.array([])
        self.y = np.array([])
        self.label = np.array([])
        self.boundary_index = np.array([])
        self.test_mode = test_mode

        self.seq_len = seq_len
        self.Ncat = 5
        # read from mat file

        self.read_mat_filelist(self.filelist)
        
        if self.shuffle:
            self.shuffle_data()

    def read_mat_filelist(self,filelist):
        """
        Scan the file list and read them one-by-one
        """
        files = []
        self.data_size = 0
        with open(filelist) as f:
            lines = f.readlines()
            for l in lines:
                items = l.split()
                files.append(items[0])
                self.data_size += int(items[1])
                print(self.data_size)
        self.X1 = np.ndarray([self.data_size, self.data_shape_1[0]])
        self.X2 = np.ndarray([self.data_size, self.data_shape_2[0], self.data_shape_2[1]])
        self.y = np.ndarray([self.data_size, self.Ncat])
        self.label = np.ndarray([self.data_size])
        count = 0
        for i in range(len(files)):
            X1, X2, y, label = self.read_mat_file(files[i].strip())
            self.X1[count : count + len(X1)] = X1
            self.X2[count : count + len(X2)] = X2
            self.y[count : count + len(X1)] = y
            self.label[count : count + len(X1)] = label
            self.boundary_index = np.append(self.boundary_index, np.arange(count, count + self.seq_len - 1))
            count += len(X1)
            print(count)

        print("Boundary indices")
        print(self.boundary_index)
        self.data_index = np.arange(len(self.X1))
        print(len(self.data_index))
        # exclude those starting indices at the end of the recordings that do not constitute full sequences
        if self.test_mode == False:
            mask = np.in1d(self.data_index,self.boundary_index, invert=True)
            self.data_index = self.data_index[mask]
            print(len(self.data_index))
        print(self.X1.shape, self.X2.shape, self.y.shape, self.label.shape)

    def read_mat_file(self,filename):
        # Load a data file
        print(filename)
        data = h5py.File(filename,'r')
        data.keys()
        # X1: raw data
        X1 = np.array(data['X1'])
        X1 = np.transpose(X1, (1, 0))  # rearrange dimension
        # X2: time-frequency data
        X2 = np.array(data['X2'])
        X2 = np.transpose(X2, (2, 1, 0))  # rearrange dimension
        y = np.array(data['y'])
        y = np.transpose(y, (1, 0))  # rearrange dimension
        label = np.array(data['label'])
        label = np.transpose(label, (1, 0))  # rearrange dimension
        label = np.squeeze(label)

        return X1, X2, y, label

        
    def shuffle_data(self):
        """
        Random shuffle the data points indexes
        """
        idx = np.random.permutation(len(self.data_index))
        self.data_index = self.data_index[idx]

                
    def reset_pointer(self):
        """
        reset pointer to begin of the list
        """
        self.pointer = 0
        
        if self.shuffle:
            self.shuffle_data()
        
    
    def next_batch(self, batch_size):
        data_index = self.data_index[self.pointer:self.pointer + batch_size]

        #update pointer
        self.pointer += batch_size

        # after stack eeg, eog, emg, data_shape now is a 3D tensor
        batch_x1 = np.ndarray([batch_size, self.seq_len, self.data_shape_1[0], self.data_shape_1[1]])
        batch_x2 = np.ndarray([batch_size, self.seq_len, self.data_shape_2[0], self.data_shape_2[1], self.data_shape_2[2]])
        batch_y = np.ndarray([batch_size, self.seq_len, self.y.shape[1]])
        batch_label = np.ndarray([batch_size, self.seq_len])

        for i in range(len(data_index)):
            for n in range(self.seq_len):
                batch_x1[i, n]  = self.X1[data_index[i] - (self.seq_len-1) + n, :, :]
                batch_x2[i, n]  = self.X2[data_index[i] - (self.seq_len-1) + n, :, :, :]
                batch_y[i, n] = self.y[data_index[i] - (self.seq_len-1) + n, :]
                batch_label[i, n] = self.label[data_index[i] - (self.seq_len-1) + n]

        batch_x1.astype(np.float32)
        batch_x2.astype(np.float32)
        batch_y.astype(np.float32)
        batch_label.astype(np.float32)

        return batch_x1, batch_x2, batch_y, batch_label

    # get and padding zeros the rest of the data if they are smaller than 1 batch
    # this necessary for testing
    def rest_batch(self, batch_size):

        data_index = self.data_index[self.pointer : len(self.data_index)]
        actual_len = len(self.data_index) - self.pointer

        # update pointer
        self.pointer = len(self.data_index)

        # after stack eeg, eog, emg, data_shape now is a 3D tensor
        batch_x1 = np.ndarray([actual_len, self.seq_len, self.data_shape_1[0], self.data_shape_1[1]])
        batch_x2 = np.ndarray([actual_len, self.seq_len, self.data_shape_2[0], self.data_shape_2[1], self.data_shape_2[2]])
        batch_y = np.ndarray([actual_len, self.seq_len, self.y.shape[1]])
        batch_label = np.ndarray([actual_len, self.seq_len])

        for i in range(len(data_index)):
            for n in range(self.seq_len):
                batch_x1[i, n]  = self.X1[data_index[i] - (self.seq_len-1) + n, :, :]
                batch_x2[i, n]  = self.X2[data_index[i] - (self.seq_len-1) + n, :, :, :]
                batch_y[i, n] = self.y[data_index[i] - (self.seq_len-1) + n, :]
                batch_label[i, n] = self.label[data_index[i] - (self.seq_len-1) + n]

        batch_x1.astype(np.float32)
        batch_x2.astype(np.float32)
        batch_y.astype(np.float32)
        batch_label.astype(np.float32)

        return actual_len, batch_x1, batch_x2, batch_y, batch_label
