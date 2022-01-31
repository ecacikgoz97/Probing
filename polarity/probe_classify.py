# -----------------------------------------------------------
# Date:        2021/12/19
# Author:      Muge Kural
# Description: Polarity Classifier for trained probe 
# -----------------------------------------------------------

import argparse, torch, json
from statistics import mode
import torch.nn as nn
from models.gpt3 import GPT3
from common.vocab import VocabEntry
from common.utils import *
from probe import MiniGPT_Probe
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   

# CONFIG
parser = argparse.ArgumentParser(description='')
args = parser.parse_args()
args.device         = device
args.mname          = 'miniGPT' 
model_path          = '/Users/emrecanacikgoz/Desktop/NLP/Probing/polarity/results/MiniGPT/4000_instances/100epochs.pt'
model_surf_vocab    = '/Users/emrecanacikgoz/Desktop/NLP/Probing/polarity/results/MiniGPT/4000_instances/surf_vocab.json'
model_polar_vocab   = '/Users/emrecanacikgoz/Desktop/NLP/Probing/polarity/results/MiniGPT/4000_instances/polar_vocab.json'

# data
with open(model_surf_vocab) as f:
    word2id = json.load(f)
    surf_vocab = VocabEntry(word2id)
with open(model_polar_vocab) as f:
    word2id = json.load(f)
    polar_vocab = VocabEntry(word2id)

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
                  expand_ratio=expand_ratio)
                  
args.embed = embed_dim
args.model = MiniGPT_Probe(args, polar_vocab)
args.model.load_state_dict(torch.load(model_path))
args.model.to(args.device)
args.model.eval()

# classify
def classify(word):
    data = [1]+ [surf_vocab[char] for char in word] + [2]
    x = torch.tensor(data).to('cpu').unsqueeze(0)
    sft = nn.Softmax(dim=2)
    # (1, 1, vocab_size)
    output_logits = args.model(x)
    probs = sft(output_logits)
    pred = torch.argmax(probs,2)[0][0].item()
    print(polar_vocab.id2word(pred))
classify('gelmedi')