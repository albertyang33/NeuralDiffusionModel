''' Define the Transformer model '''
import torch
import torch.nn as nn
import numpy as np
import transformer.Constants as Constants
from transformer.Modules import BottleLinear as Linear
from transformer.Layers import EncoderLayer, DecoderLayer

def get_attn_padding_mask(seq_q, seq_k):
    ''' Indicate the padding-related part to mask '''
    assert seq_q.dim() == 2 and seq_k.dim() == 2
    mb_size, len_q = seq_q.size()
    mb_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(Constants.PAD).unsqueeze(1)   # bx1xsk
    pad_attn_mask = pad_attn_mask.expand(mb_size, len_q, len_k) # bxsqxsk
    return pad_attn_mask

def get_attn_subsequent_mask(seq):
    ''' Get an attention mask to avoid using the subsequent info.'''
    assert seq.dim() == 2
    attn_shape = (seq.size(0), seq.size(1), seq.size(1))
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    subsequent_mask = torch.from_numpy(subsequent_mask)
    if seq.is_cuda:
        subsequent_mask = subsequent_mask.cuda()
    return subsequent_mask

def get_previous_user_mask(seq, user_size):
    ''' Mask previous activated users.'''
    assert seq.dim() == 2
    prev_shape = (seq.size(0), seq.size(1), seq.size(1))
    seqs = seq.repeat(1,1,seq.size(1)).view(seq.size(0), seq.size(1), seq.size(1))
    previous_mask = np.tril(np.ones(prev_shape)).astype('float32')
    previous_mask = torch.from_numpy(previous_mask)
    if seq.is_cuda:
        previous_mask = previous_mask.cuda()
    #print(previous_mask)
    #print(seqs)
    masked_seq = previous_mask * seqs.data.float()
    #print(masked_seq.size())

    # force the 0th dimension (PAD) to be masked
    PAD_tmp = torch.zeros(seq.size(0), seq.size(1), 1)
    if seq.is_cuda:
        PAD_tmp = PAD_tmp.cuda()
    masked_seq = torch.cat([masked_seq,PAD_tmp],dim=2)
    ans_tmp = torch.zeros(seq.size(0), seq.size(1), user_size)
    if seq.is_cuda:
        ans_tmp = ans_tmp.cuda()
    masked_seq = ans_tmp.scatter_(2,masked_seq.long(),float('-inf'))
    
    return masked_seq

class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''
    def __init__(
            self, user_size, kernel_size=3, n_layers=1, n_head=1, d_k=32, d_v=32,
            d_word_vec=32, d_model=32, d_inner_hid=32, dropout=0.1, finit=0):

        super(Decoder, self).__init__()
        self.d_model = d_model
        self.user_size = user_size

        self.user_emb = nn.Embedding(
            user_size, d_word_vec, padding_idx=Constants.PAD)
        self.tgt_user_proj = Linear(d_model, user_size, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.conv = nn.Conv1d(d_model, user_size, kernel_size, padding=kernel_size-1, bias=True)
        self.padding = kernel_size-1
        self.finit = finit

        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, tgt_seq, return_attns=False, generate=False):
        if not generate:
            tgt_seq = tgt_seq[:, :-1]
        # Word embedding look up
        dec_input = self.user_emb(tgt_seq)

        # Decode
        dec_slf_attn_pad_mask = get_attn_padding_mask(tgt_seq, tgt_seq)
        dec_slf_attn_sub_mask = get_attn_subsequent_mask(tgt_seq)
        # 1 means masked
        dec_slf_attn_mask = torch.gt(dec_slf_attn_pad_mask + dec_slf_attn_sub_mask, 0)


        if return_attns:
            dec_slf_attns = []

        dec_output = dec_input
        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn = dec_layer(
                dec_output, slf_attn_mask=dec_slf_attn_mask)

            if return_attns:
                dec_slf_attns += [dec_slf_attn]

        #print(dec_output.size())
        dec_output = dec_output.transpose(1,2)
        dec_output = self.conv(dec_output)
        dec_output = dec_output[:,:,0:-self.padding]
        dec_output = dec_output.transpose(1,2).contiguous()
        if self.finit > 0:
             dec_output += self.tgt_user_proj(self.user_emb(tgt_seq[:,0])).repeat(dec_input.size(1),1,1).transpose(0,1).contiguous()

        seq_logit =  dec_output + torch.autograd.Variable(get_previous_user_mask(tgt_seq, self.user_size),requires_grad=False)
        #print(seq_logit.size()) batch*seqlen*n_word
        if return_attns:
            return seq_logit.view(-1, seq_logit.size(2)), dec_slf_attns
        else:
            return seq_logit.view(-1, seq_logit.size(2)),
