'''
This script handling the training process.
'''

import argparse
import math
import time

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import transformer.Constants as Constants
from transformer.Models import Decoder
from transformer.Optim import ScheduledOptim
from DataLoader import DataLoader

def get_performance(crit, pred, gold, smoothing=False, num_class=None):
    ''' Apply label smoothing if needed '''

    # TODO: Add smoothing
    if smoothing:
        assert bool(num_class)
        eps = 0.1
        gold = gold * (1 - eps) + (1 - gold) * eps / num_class
        raise NotImplementedError

    loss = crit(pred, gold.contiguous().view(-1))

    pred = pred.max(1)[1]

    gold = gold.contiguous().view(-1)
    n_correct = pred.data.eq(gold.data)
    n_correct = n_correct.masked_select(gold.ne(Constants.PAD).data).sum()

    return loss, n_correct

def train_epoch(model, training_data, crit, optimizer):
    ''' Epoch operation in training phase'''

    model.train()

    total_loss = 0
    n_total_words = 0
    n_total_correct = 0

    for batch in tqdm(
            training_data, mininterval=2,
            desc='  - (Training)   ', leave=False):

        # prepare data
        tgt = batch
        gold = tgt[:, 1:]

        # forward
        optimizer.zero_grad()
        pred, *_ = model(tgt)

        # backward
        loss, n_correct = get_performance(crit, pred, gold)
        loss.backward()

        # update parameters
        optimizer.step()
        optimizer.update_learning_rate()

        # note keeping
        n_words = gold.data.ne(Constants.PAD).sum()
        n_total_words += n_words
        n_total_correct += n_correct
        total_loss += loss.data[0]

    return total_loss/n_total_words, n_total_correct/n_total_words

def train(model, training_data, crit, optimizer, opt):
    ''' Start training '''

    for epoch_i in range(opt.epoch):
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        train_loss, train_accu = train_epoch(model, training_data, crit, optimizer)
        print('  - (Training) accuracy: {accu:3.3f} %, '\
              'elapse: {elapse:3.3f} min'.format(
                   accu=100*train_accu,
                  elapse=(time.time()-start)/60))

        model_state_dict = model.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'settings': opt,
            'epoch': epoch_i}

        if opt.save_model:
            model_name = opt.save_model + str(epoch_i)+'.chkpt'
            torch.save(checkpoint, model_name)


def main():
    ''' Main function'''
    parser = argparse.ArgumentParser()

    parser.add_argument('-epoch', type=int, default=20)
    parser.add_argument('-batch_size', type=int, default=32)

    parser.add_argument('-d_model', type=int, default=64)
    parser.add_argument('-d_inner_hid', type=int, default=64)
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-d_v', type=int, default=64)
    parser.add_argument('-window_size',type=int, default=3)
    parser.add_argument('-finit',type=int, default=0)

    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_warmup_steps', type=int, default=1000)

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-embs_share_weight', action='store_true')
    parser.add_argument('-proj_share_weight', action='store_true')

    parser.add_argument('-save_model', default='Lastfm_test')
    parser.add_argument('-no_cuda', action='store_true')

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    opt.d_word_vec = opt.d_model
    
    #========= Preparing DataLoader =========#
    train_data = DataLoader(use_valid=False, load_dict=True, batch_size=opt.batch_size, cuda=opt.cuda)
    opt.user_size = train_data.user_size

    #========= Preparing Model =========#

    decoder = Decoder(
        opt.user_size,
        d_k=opt.d_k,
        d_v=opt.d_v,
        d_model=opt.d_model,
        d_word_vec=opt.d_word_vec,
        d_inner_hid=opt.d_inner_hid,
        n_head=opt.n_head,
        kernel_size=opt.window_size,
        dropout=opt.dropout)

    optimizer = ScheduledOptim(
        optim.Adam(
            decoder.parameters(),
            betas=(0.9, 0.98), eps=1e-09),
        opt.d_model, opt.n_warmup_steps)


    def get_criterion(user_size):
        ''' With PAD token zero weight '''
        weight = torch.ones(user_size)
        weight[Constants.PAD] = 0
        weight[Constants.EOS] = 0
        return nn.CrossEntropyLoss(weight, size_average=False)

    crit = get_criterion(train_data.user_size)

    if opt.cuda:
        decoder = decoder.cuda()
        crit = crit.cuda()

    train(decoder, train_data, crit, optimizer, opt)

if __name__ == '__main__':
    main()
