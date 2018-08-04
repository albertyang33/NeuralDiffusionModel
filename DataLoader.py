''' Data Loader class for training iteration '''
import random
import numpy as np
import torch
from torch.autograd import Variable
import transformer.Constants as Constants
import logging
import pickle
class Options(object):
    
    def __init__(self):
        #data options.

        #train file path.
        self.train_data = 'data/lastfm/cascade.txt'

        #test file path.
        self.test_data = 'data/lastfm/cascadetest.txt'

        self.u2idx_dict = 'data/lastfm/u2idx.pickle'

        self.idx2u_dict = 'data/lastfm/idx2u.pickle'
        #save path.
        self.save_path = ''

        self.batch_size = 32

class DataLoader(object):
    ''' For data iteration '''

    def __init__(
            self, use_valid=False, load_dict=True, cuda=True, batch_size=32, shuffle=True, test=False):
        self.options = Options()
        self.options.batch_size = batch_size
        self._u2idx = {}
        self._idx2u = []
        self.use_valid = use_valid
        if not load_dict:
            self._buildIndex()
            with open(self.options.u2idx_dict, 'wb') as handle:
                pickle.dump(self._u2idx, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(self.options.idx2u_dict, 'wb') as handle:
                pickle.dump(self._idx2u, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(self.options.u2idx_dict, 'rb') as handle:
                self._u2idx = pickle.load(handle)
            with open(self.options.idx2u_dict, 'rb') as handle:
                self._idx2u = pickle.load(handle)
            self.user_size = len(self._u2idx)

        self._train_cascades = self._readFromFile(self.options.train_data)
        self._test_cascades = self._readFromFile(self.options.test_data)
        self.train_size = len(self._train_cascades)
        self.test_size = len(self._test_cascades)
        print("user size:%d" % (self.user_size-2)) # minus pad and eos
        print("training set size:%d    testing set size:%d" % (self.train_size, self.test_size))

        self.cuda = cuda
        self.test = test
        if not self.use_valid:
            self._n_batch = int(np.ceil(len(self._train_cascades) / batch_size))
        else:
            self._n_batch = int(np.ceil(len(self._test_cascades) / batch_size))

        self._batch_size = self.options.batch_size

        self._iter_count = 0

        self._need_shuffle = shuffle

        if self._need_shuffle:
            random.shuffle(self._train_cascades)

    def _buildIndex(self):
        #compute an index of the users that appear at least once in the training and testing cascades.
        opts = self.options

        train_user_set = set()
        test_user_set = set()

        lineid=0
        for line in open(opts.train_data):
            lineid+=1
            if len(line.strip()) == 0:
                continue
            chunks = line.strip().split()
            for chunk in chunks:
                try:
                    user, timestamp = chunk.split(',')
                except:
                    print(line)
                    print(chunk)
                    print(lineid)
                train_user_set.add(user)

        for line in open(opts.test_data):
            if len(line.strip()) == 0:
                continue
            chunks = line.strip().split()
            for chunk in chunks:
                user, timestamp = chunk.split(',')
                test_user_set.add(user)

        user_set = train_user_set | test_user_set

        pos = 0
        self._u2idx['<blank>'] = pos
        self._idx2u.append('<blank>')
        pos += 1
        self._u2idx['</s>'] = pos
        self._idx2u.append('</s>')
        pos += 1

        for user in user_set:
            self._u2idx[user] = pos
            self._idx2u.append(user)
            pos += 1
        opts.user_size = len(user_set) + 2
        self.user_size = len(user_set) + 2
        print("user_size : %d" % (opts.user_size))


    def _readFromFile(self, filename):
        """read all cascade from training or testing files. """
        t_cascades = []
        for line in open(filename):
            if len(line.strip()) == 0:
                continue
            userlist = []
            chunks = line.strip().split()
            for chunk in chunks:
                try:
                    user, timestamp = chunk.split(',')
                except:
                    print(chunk)
                if user in self._u2idx:
                    userlist.append(self._u2idx[user])
                    #if len(userlist) > 500:
                    #    break
                    # uncomment these lines if your GPU memory is not enough

            if len(userlist) > 1:
                userlist.append(Constants.EOS)
                t_cascades.append(userlist)
        return t_cascades

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self._n_batch

    def next(self):
        ''' Get the next batch '''

        def pad_to_longest(insts):
            ''' Pad the instance to the max seq length in batch '''

            max_len = max(len(inst) for inst in insts)

            inst_data = np.array([
                inst + [Constants.PAD] * (max_len - len(inst))
                for inst in insts])
        
            inst_data_tensor = Variable(
                torch.LongTensor(inst_data), volatile=self.test)

            if self.cuda:
                inst_data_tensor = inst_data_tensor.cuda()

            return inst_data_tensor

        if self._iter_count < self._n_batch:
            batch_idx = self._iter_count
            self._iter_count += 1

            start_idx = batch_idx * self._batch_size
            end_idx = (batch_idx + 1) * self._batch_size

            if not self.use_valid:
                seq_insts = self._train_cascades[start_idx:end_idx]
            else:
                seq_insts = self._test_cascades[start_idx:end_idx]
            seq_data = pad_to_longest(seq_insts)
            #print('???')
            #print(seq_data.data)
            #print(seq_data.size())
            return seq_data
        else:

            if self._need_shuffle:
                random.shuffle(self._train_cascades)
                #random.shuffle(self._test_cascades)

            self._iter_count = 0
            raise StopIteration()
