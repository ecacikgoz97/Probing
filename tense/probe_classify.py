# -----------------------------------------------------------
# Date:        2021/12/19
# Author:      Muge Kural
# Description: Tense Classifier for trained probe 
# -----------------------------------------------------------

import sys, argparse, random, torch, json, matplotlib, os
import torch.nn as nn
import numpy as np
from models.gpt3 import GPT3
from common.vocab import VocabEntry
from common.utils import *
from probe import MiniGPT_Probe
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   

# CONFIG
parser = argparse.ArgumentParser(description='')
args = parser.parse_args()
args.device         = 'cuda'
args.mname          = 'ae_probe' 
model_path          = 'evaluation/probing/tense/results/training/ae_1_probe/4000_instances/200epochs.pt'
model_surf_vocab    = 'evaluation/probing/tense/results/training/ae_1_probe/4000_instances/surf_vocab.json'
model_tense_vocab   = 'evaluation/probing/tense/results/training/ae_1_probe/4000_instances/tense_vocab.json'

# data
with open(model_surf_vocab) as f:
    word2id = json.load(f)
    surf_vocab = VocabEntry(word2id)
with open(model_tense_vocab) as f:
    word2id = json.load(f)
    tense_vocab = VocabEntry(word2id)

# model
model_init = uniform_initializer(0.01); emb_init = uniform_initializer(0.1)
args.ni = 512; args.nz = 32; 
args.enc_nh = 1024; args.dec_nh = 1024
args.enc_dropout_in = 0.0; args.enc_dropout_out = 0.0
args.dec_dropout_in = 0.0; args.dec_dropout_out = 0.0
args.pretrained_model = AE(args, surf_vocab, model_init, emb_init)
args.nh = args.enc_nh
args.model = AE_Probe(args, tense_vocab, model_init, emb_init)
args.model.load_state_dict(torch.load(model_path))
args.model.to(args.device)
args.model.eval()

# classify
def classify(word):
    data = [1]+ [surf_vocab[char] for char in word] + [2]
    x = torch.tensor(data).to('cuda').unsqueeze(0)
    sft = nn.Softmax(dim=2)
    # (1, 1, vocab_size)
    output_logits = args.model(x)
    probs = sft(output_logits)
    pred = torch.argmax(probs,2)[0][0].item()
    print(tense_vocab.id2word(pred))
classify('gidiyorum')