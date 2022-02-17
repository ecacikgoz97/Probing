# -----------------------------------------------------------
# Date:        2021/12/19 
# Author:      Muge Kural
# Description: Trainer of surface form pos tagging probe, saves the results under ./results directory.
# -----------------------------------------------------------

import sys, argparse, random, torch, json, matplotlib, os
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import optim
from common.utils import *
from data.data import build_data, log_data
from models.gpt3 import GPT3
from common.vocab import VocabEntry
from probe import MiniGPT_Probe, MiniGPT_Probe2
matplotlib.use('Agg')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
local_set = False
if local_set == True:
    working_path = "/Users/emrecanacikgoz/Desktop/"
else:
    working_path = "/kuacc/users/eacikgoz17/"  

def test(batches, mode, args):
    epoch_loss = 0; epoch_acc = 0;  epoch_num_instances = 0
    numbatches = len(batches)
    indices = list(range(numbatches))
    for i, idx in enumerate(indices):
        # (batchsize, t)
        surf, surfpos = batches[idx] 
        loss, acc = args.model.probe_loss(surf, surfpos)
        epoch_num_instances += surf.size(0)
        epoch_loss += loss.item()
        epoch_acc  += acc
    nll = epoch_loss / numbatches
    acc = epoch_acc / epoch_num_instances
    args.logger.write('%s --- avg_loss: %.4f, acc: %.4f  \n' % (mode, nll, acc))
    return nll,  acc

def train(data, args):
    trnbatches, valbatches, tstbatches = data
    opt = optim.Adam(filter(lambda p: p.requires_grad, args.model.parameters()), lr=args.lr)
    scheduler = ReduceLROnPlateau(opt, 'min', verbose=1, factor=0.5)
    for name, prm in args.model.named_parameters():
        args.logger.write('\n'+name+', '+str(prm.shape) + ': '+ str(prm.requires_grad))
    numbatches = len(trnbatches)
    indices = list(range(numbatches))
    random.seed(0)
    best_loss = 1e4
    trn_loss_values = []; trn_acc_values = []
    val_loss_values = []; val_acc_values = []
    for epc in range(args.epochs):
        epoch_loss = 0; epoch_acc = 0; epoch_num_instances = 0
        random.shuffle(indices) # this breaks continuity if there is
        for i, idx in enumerate(indices):
            args.model.zero_grad() 
            # (batchsize, t)
            surf, surfpos = trnbatches[idx]
            loss, acc = args.model.probe_loss(surf, surfpos)
            loss.backward()
            opt.step()
            epoch_num_instances += surf.size(0) 
            epoch_loss       += loss.item()
            epoch_acc        += acc
        nll = epoch_loss / numbatches
        acc = epoch_acc / epoch_num_instances
        trn_loss_values.append(nll)
        trn_acc_values.append(acc)
        args.logger.write('\nepoch: %.1d avg_loss: %.4f, acc: %.4f \n' % (epc, nll, acc))
        # VAL
        args.model.eval()
        with torch.no_grad():
            nll, acc = test(valbatches, "val", args)
        val_loss_values.append(nll)
        val_acc_values.append(acc)
        #scheduler.step(nll)
        if nll < best_loss:
            args.logger.write('update best loss \n')
            best_loss = nll
            torch.save(args.model.state_dict(), args.save_path)
        args.model.train()
    plot_curves(args.task, args.mname, args.fig, args.axs[0], trn_loss_values, val_loss_values, args.plt_style, 'loss')
    plot_curves(args.task, args.mname, args.fig, args.axs[1], trn_acc_values,  val_acc_values,  args.plt_style, 'acc')
    

# CONFIG
parser = argparse.ArgumentParser(description='')
args = parser.parse_args()
args.device = device 
args.mname  = 'MiniGPT_1000epochs_lr0001' 
model_path  = working_path + 'NLP/EXPERIMENTS/exp14/charlm_miniGPT/results/50000_instances500epochs.pt'
model_vocab = working_path + 'NLP/EXPERIMENTS/exp14/charlm_miniGPT/results/surf_vocab.json'

# training
args.batchsize = 128; args.epochs = 1000
args.opt= 'Adam'; args.lr = 0.0001
args.task = 'surf2surfpos'
args.seq_to_no_pad = 'surface'

# data
with open(model_vocab) as f:
    word2id = json.load(f)
    surf_vocab = VocabEntry(word2id)
args.trndata = working_path + 'NLP/Probing/pos_tagging/data/surfpos.uniquesurfs.trn.txt' 
args.valdata = working_path + 'NLP/Probing/pos_tagging/data/surfpos.uniquesurfs.val.txt'
args.tstdata = working_path + 'NLP/Probing/pos_tagging/data/surfpos.uniquesurfs.val.txt' 
args.maxtrnsize = 57769; args.maxvalsize = 10000; args.maxtstsize = 10000
rawdata, batches, vocab = build_data(args, surf_vocab)
_, surfpos_vocab  = vocab
trndata, vlddata, tstdata = rawdata
args.trnsize , args.valsize, args.tstsize = len(trndata), len(vlddata), len(tstdata)

# model
num_layers=3
embed_dim=128
num_heads=16
block_size=128
embedding_dropout_rate=0.15 
attention_dropout_rate=0.15
residual_dropout_rate=0.15
expand_ratio = 4
args.pretrained_model = GPT3(vocab=surf_vocab,
                             num_layers=num_layers,
                             embed_dim=embed_dim,
                             num_heads=num_heads,
                             block_size=block_size,
                             embedding_dropout_rate=embedding_dropout_rate,
                             attention_dropout_rate=attention_dropout_rate,
                             residual_dropout_rate=residual_dropout_rate,
                             expand_ratio=expand_ratio
                            )
args.pretrained_model.load_state_dict(torch.load(model_path))
args.embed = embed_dim
args.model = MiniGPT_Probe(args, surfpos_vocab)
print(args.model)
for param in args.model.token_embedding.parameters():
    param.requires_grad = False
for param in args.model.decoder1.parameters():
    param.requires_grad = False
for param in args.model.decoder2.parameters():
    param.requires_grad = False
for param in args.model.decoder3.parameters():
    param.requires_grad = False
args.model.to(args.device)
print(args.model)

# logging
args.modelname = working_path + 'NLP/Probing/pos_tagging/results/'+args.mname+'/'+str(len(trndata))+'_instances/'
try:
    os.makedirs(args.modelname)
    print("Directory " , args.modelname ,  " Created ") 
except FileExistsError:
    print("Directory " , args.modelname ,  " already exists")
args.save_path = args.modelname +  str(args.epochs)+'epochs.pt'
args.log_path =  args.modelname +  str(args.epochs)+'epochs.log'
args.fig_path =  args.modelname +  str(args.epochs)+'epochs.png'
args.logger = Logger(args.log_path)
with open(args.modelname+'/surf_vocab.json', 'w') as f:
    f.write(json.dumps(surf_vocab.word2id))
with open(args.modelname+'/surfpos_vocab.json', 'w') as f:
    f.write(json.dumps(surfpos_vocab.word2id))
args.logger.write('\nnumber of params: %d \n' % count_parameters(args.model))
args.logger.write(args)
args.logger.write('\n')

# plotting
args.fig, args.axs = plt.subplots(2, sharex=True)
args.plt_style = pstyle = '-'

# run
train(batches, args)
plt.savefig(args.fig_path)


  