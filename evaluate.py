''' Translate input text with trained model. '''

import torch
import random
import argparse
from tqdm import tqdm
import numpy as np
from transformer.Translator import Translator
import transformer.Constants as Constants
from DataLoader import DataLoader

def getF1(ground_truth, pred_cnt):
    right = np.dot(ground_truth, pred_cnt)
    pred = np.sum(pred_cnt)
    total = np.sum(ground_truth)
    if pred == 0:
        return 0, 0, 0
    precision = right / pred
    recall = right / total
    if precision==0 or recall ==0:
        return 0, 0, 0
    return (2*precision*recall) / (precision+recall), precision, recall

def getMSE(ground_truth, pred_cnt):
    return mean_squared_error(ground_truth, pred_cnt)

def getMAP(ground_truth, pred_cnt):
    size_cascade = list(ground_truth).count(1)
    avgp = 0.
    ite_list = [[idx, item, ground_truth[idx]] for idx, item in enumerate(pred_cnt)]
    ite_list = sorted(ite_list, key= lambda x:x[1], reverse=True)
    n_positive = 0
    idx = 0
    while n_positive < size_cascade:
        if ite_list[idx][2] == 1:
            n_positive += 1
            pre = n_positive / (idx + 1.)
            avgp = avgp + pre
        idx += 1
        #if idx >= 1:
        #    break
    avgp = avgp / size_cascade
    return avgp

def main():
    '''Main Function'''

    parser = argparse.ArgumentParser(description='translate.py')

    parser.add_argument('-model', required=True,
                        help='Path to model .pt file')
    '''
    parser.add_argument('-src', required=True,
                        help='Source sequence to decode (one line per sequence)')
    parser.add_argument('-vocab', required=True,
                        help='Source sequence to decode (one line per sequence)')
    '''
    parser.add_argument('-output', default='pred.txt',
                        help="""Path to output the predictions (each line will
                        be the decoded sequence""")
    parser.add_argument('-beam_size', type=int, default=100,
                        help='Beam size')
    parser.add_argument('-batch_size', type=int, default=1,
                        help='Batch size')
    parser.add_argument('-n_best', type=int, default=1,
                        help="""If verbose is set, will output the n_best
                        decoded sentences""")
    parser.add_argument('-no_cuda', action='store_true')

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda

    # Prepare DataLoader

    test_data = DataLoader(use_valid=True, batch_size=opt.batch_size, cuda=opt.cuda)

    translator = Translator(opt)
    translator.model.eval()

    numuser = test_data.user_size

    num_right=0
    num_total=0

    avgF1 = 0
    avgPre = 0
    avgRec = 0

    avgF1_long = 0
    avgPre_long = 0
    avgRec_long = 0

    avgF1_short = 0
    avgPre_short = 0
    avgRec_short = 0
    numseq = 0 # number of test seqs

    # for micro pre rec f1
    right = 0.
    pred = 0.
    total = 0.
    right_long = 0.
    pred_long = 0.
    total_long = 0.
    right_short = 0.
    pred_short = 0.
    total_short = 0.

    with open(opt.output, 'w') as f:
        for batch in tqdm(test_data, mininterval=2, desc='  - (Test)', leave=False):
            all_samples = translator.translate_batch(batch).data

            for bid in range(batch.size(0)):
                numseq += 1.0

                ground_truth = np.zeros([numuser])
                num_ground_truth = 0
                for user in batch.data[bid][1:-1]:
                    if user == Constants.EOS or user==Constants.PAD:
                        break
                    ground_truth[user]=1.0
                    num_ground_truth += 1
                
                pred_cnt = np.zeros([numuser])
                for beid in range(opt.beam_size):
                    for pred_uid in all_samples[bid,beid,1:num_ground_truth+1]:
                        if pred_uid == Constants.EOS:
                            break
                        else:
                            pred_cnt[pred_uid] += 1.0 / opt.beam_size
                
                F1,pre,rec = getF1(ground_truth, pred_cnt)
                avgF1 += F1
                avgPre += pre
                avgRec += rec
                right += np.dot(ground_truth, pred_cnt)
                pred += np.sum(pred_cnt)
                total += np.sum(ground_truth) 

                # for short user
                ground_truth = np.zeros([numuser])
                num_ground_truth = 0
                for user in batch.data[bid][1:-1]:
                    if user == Constants.EOS or user==Constants.PAD:
                        break
                    ground_truth[user]=1.0
                    num_ground_truth += 1
                    if num_ground_truth >= 5:
                        break
                
                pred_cnt = np.zeros([numuser])
                for beid in range(opt.beam_size):
                    #total += len(ground_truth)
                    for pred_uid in all_samples[bid,beid,1:num_ground_truth+1]:
                        if pred_uid == Constants.EOS:
                            break
                            #continue
                        else:
                            pred_cnt[pred_uid] += 1.0 / opt.beam_size
                
                F1,pre,rec = getF1(ground_truth, pred_cnt)
                avgF1_short += F1
                avgPre_short += pre
                avgRec_short += rec
                right_short += np.dot(ground_truth, pred_cnt)
                pred_short += np.sum(pred_cnt)
                total_short += np.sum(ground_truth)

    print('[Info] Finished.')
    print('Macro')
    print(avgF1 / numseq)
    print(avgPre / numseq)
    print(avgRec / numseq)
    print('Results for the first no more than 5 predictions')
    print(avgF1_short / numseq)
    print(avgPre_short / numseq)
    print(avgRec_short / numseq)

    print('Micro')
    pmi = right / pred
    rmi = right / total
    print(2*pmi*rmi/(pmi+rmi))
    print(pmi)
    print(rmi)

    print('Results for the first no more than 5 predictions')
    pmi_long = right_short / pred_short
    rmi_long = right_short / total_short
    print(2*pmi_long*rmi_long/(pmi_long+rmi_long))
    print(pmi_long)
    print(rmi_long)

if __name__ == "__main__":
    main()
