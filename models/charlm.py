"""
        1) x.shape: torch.Size([128, 15])
        2) src.shape: torch.Size([128, 14])
        3) word_embed.shape: torch.Size([128, 14, 512])
        4) word_embed.shape: torch.Size([128, 14, 512])
        5) output.shape: torch.Size([128, 14, 1024]), last_state: torch.Size([1, 128, 1024]), last_cell: torch.Size([1, 128, 1024])
        6) output.shape: torch.Size([128, 14, 1024])
        7) output_logits.shape: torch.Size([128, 14, 39])
        8) _tgt.shape: torch.Size([1792])
        9) _output_logits.shape: torch.Size([1792, 39])
        "[B, T, C]: for each T, C number of scores"

        1) x.shape: torch.Size([128, 6])
        2) src.shape: torch.Size([128, 5])
        3) word_embed.shape: torch.Size([128, 5, 512])
        4) word_embed.shape: torch.Size([128, 5, 512])
        5) output.shape: torch.Size([128, 5, 1024]), last_state: torch.Size([1, 128, 1024]), last_cell: torch.Size([1, 128, 1024])
        6) output.shape: torch.Size([128, 5, 1024])
        7) output_logits.shape: torch.Size([128, 5, 39])
        8) _tgt.shape: torch.Size([640])
        9) _output_logits.shape: torch.Size([640, 39])
"""

import torch
import torch.nn as nn

class CharLM(nn.Module):
    """docstring for CharLM"""
    def __init__(self, args, vocab, model_init, emb_init, bidirectional=False):
        super(CharLM, self).__init__()
        self.ni = args.ni
        self.nh = args.nh
        self.vocab = vocab

        self.embed = nn.Embedding(len(vocab.word2id), args.ni, padding_idx=0)
        self.lstm = nn.LSTM(input_size=args.ni, hidden_size=args.nh, num_layers=1, batch_first=True, dropout=0, bidirectional=bidirectional)

        self.dropout_in = nn.Dropout(args.enc_dropout_in)
        self.dropout_out = nn.Dropout(args.enc_dropout_out)

        # prediction layer
        self.pred_linear = nn.Linear(args.nh, len(vocab.word2id), bias=False)
        vocab_mask = torch.ones(len(vocab.word2id))
        self.loss = nn.CrossEntropyLoss(weight=vocab_mask, reduce=False, ignore_index=0)
        self.reset_parameters(model_init, emb_init)

    def reset_parameters(self, model_init, emb_init):
        for param in self.parameters():
            model_init(param)
        emb_init(self.embed.weight)


    def forward(self, x):
        #print(f"1) x.shape: {x.shape}")
        src = x[:, :-1] # remove end symbol
        tgt = x[:, 1:]  # remove start symbol
        batch_size, seq_len = src.size()

        # (batch_size, seq_len-1, args.ni)
        #print(f"2) src.shape: {src.shape}")
        word_embed = self.embed(src)
        #print(f"3) word_embed.shape: {word_embed.shape}")
        word_embed = self.dropout_in(word_embed)
        #print(f"4) word_embed.shape: {word_embed.shape}")

        output, (last_state, last_cell) = self.lstm(word_embed)
        #print(f"5) output.shape: {output.shape}, last_state: {last_state.shape}, last_cell: {last_cell.shape}")
        output = self.dropout_out(output)
        #print(f"6) output.shape: {output.shape}")
        # (batch_size, seq_len, vocab_size)
        output_logits = self.pred_linear(output)
        #print(f"7) output_logits.shape: {output_logits.shape}")
       
       # (batch_size * seq_len)
        _tgt = tgt.contiguous().view(-1)
        #print(f"8) _tgt.shape: {_tgt.shape}")

        # (batch_size * seq_len, vocab_size)
        _output_logits = output_logits.view(-1, output_logits.size(2))
        #print(f"9) _output_logits.shape: {_output_logits.shape}\n\n\n")

        # (batch_size * seq_len)
        loss = self.loss(_output_logits,  _tgt)

        # loss: (batch_size): sum the loss over sequence
        return loss.view(batch_size, seq_len).sum(-1), self.accuracy(output_logits, tgt), output_logits



    def accuracy(self, output_logits, targets):
        # output_logits: (B, T, vocab_size), targets: (B,T)
        surface_vocab = self.vocab
        B, T = targets.size()
        sft = nn.Softmax(dim=2)
        # (batchsize, T)
        pred_tokens = torch.argmax(sft(output_logits),2)
        correct_tokens = (pred_tokens == targets)
        wrong_tokens = (pred_tokens != targets)
        wrong_predictions = []
        correct_predictions = []
        for i in range(B):
            target  = ''.join(surface_vocab.decode_sentence(targets[i]))
            pred = ''.join(surface_vocab.decode_sentence(pred_tokens[i]))
            if target != pred:
                wrong_predictions.append('target: %s pred: %s' % (target, pred))
            else:
                correct_predictions.append('target: %s pred: %s' % (target, pred))
        acc = correct_tokens.sum().item(), B*T, wrong_tokens.sum().item(), wrong_predictions, correct_predictions
        return  acc
