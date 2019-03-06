import numpy as np
# import tensorflow as tf

class Dataset:
    def __init__(self,data):
        self._index_in_epoch = 0
        self._epochs_completed = 0
        X, y = data
        self._data = None
        if(y is None):
            self._data = X
        else
            self._data = np.concatenate([X, y.reshape(len(y),1)], axis=1)
        self._num_examples = self._data.shape[0]

    @property
    def data(self):
        return self._data


    def get_next_batch(self,batch_size,shuffle = True):
        start = self._index_in_epoch
        if start == 0 and self._epochs_completed == 0:
            idx = np.arange(0, self._num_examples)  # get all possible indexes
            np.random.shuffle(idx)  # shuffle indices
            self._data = self._data[idx]  # get list of `num` random samples

        # go to the next batch
        if start + batch_size > self._num_examples:
            self._epochs_completed += 1
            rest_num_examples = self._num_examples - start
            data_rest_part = self._data[start:self._num_examples]
            idx0 = np.arange(0, self._num_examples)  # get all possible indexes
            np.random.shuffle(idx0)  # shuffle indexes
            self._data = self._data[idx0] # get list of `num` random samples

            start = 0
            self._index_in_epoch = batch_size - rest_num_examples #avoid the case where the #sample != integer times of batch_size
            end =  self._index_in_epoch  
            data_new_part =  self._data[start:end]  
            return np.concatenate([data_rest_part, data_new_part], axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._data[start:end]
            
# dataset = Dataset(np.arange(0, 10))
# for i in range(10):
#     print(dataset.get_next_batch(5))