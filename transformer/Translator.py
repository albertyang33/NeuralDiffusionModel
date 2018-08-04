''' This module will handle the text generation with beam search. '''

import torch
import torch.nn as nn
from torch.autograd import Variable

from transformer.Models import Decoder

class Translator(object):
    ''' Load with trained model and handle the beam search '''

    def __init__(self, opt):
        self.opt = opt
        self.tt = torch.cuda if opt.cuda else torch

        checkpoint = torch.load(opt.model)
        model_opt = checkpoint['settings']
        self.model_opt = model_opt

        model = Decoder(
        model_opt.user_size,
        d_k=model_opt.d_k,
        d_v=model_opt.d_v,
        d_model=model_opt.d_model,
        d_word_vec=model_opt.d_word_vec,
        kernel_size=model_opt.window_size,
        finit=model_opt.finit,
        d_inner_hid=model_opt.d_inner_hid,
        n_head=model_opt.n_head,
        dropout=model_opt.dropout)

        prob_projection = nn.Softmax()

        model.load_state_dict(checkpoint['model'])
        print('[Info] Trained model state loaded.')

        if opt.cuda:
            model.cuda()
            prob_projection.cuda()
        else:
            model.cpu()
            prob_projection.cpu()

        model.prob_projection = prob_projection

        self.model = model
        self.model.eval()

    def translate_batch(self, batch):
        ''' Translation work in one batch '''

        # Batch size is in different location depending on data.
        tgt_seq = batch
        #print(batch.size())

        batch_size = tgt_seq.size(0)
        max_len = min(tgt_seq.size(1), 100)
        beam_size = self.opt.beam_size

        #- Decode
        #print(tgt_seq.data[0,:])
        dec_partial_seq = torch.stack([
                torch.LongTensor(beam_size).fill_(tgt_seq.data[j,0]).unsqueeze(1) for j in range(batch_size)])
        # size: (batch * beam) x seq
        
        # wrap into a Variable
        dec_partial_seq = Variable(dec_partial_seq, volatile=True)
        if self.opt.cuda:
            dec_partial_seq = dec_partial_seq.cuda()
        for i in range(max_len+1):

            len_dec_seq = i+1
            dec_partial_seq = dec_partial_seq.view(-1, len_dec_seq)

            # -- Decoding -- #
            dec_output, *_ = self.model(dec_partial_seq, generate=True)
            dec_output = dec_output.view(dec_partial_seq.size(0),-1,self.model_opt.user_size)
            dec_output = dec_output[:, -1, :] # (batch * beam) * user_size
            out = self.model.prob_projection(dec_output)
            sample = torch.multinomial(out, 1, replacement=True)

            # batch x beam x 1
            #sample = sample.view(batch_size, beam_size, 1).contiguous()
            sample = sample.long()
            dec_partial_seq = torch.cat([dec_partial_seq, sample], dim=1)
            dec_partial_seq = dec_partial_seq.view(batch_size, beam_size,-1)
            #print(dec_partial_seq.size())

 

        #- Return useful information
        return dec_partial_seq
