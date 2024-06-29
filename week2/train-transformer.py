from __future__ import print_function
import argparse
import random
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformer import TransformerEncoder, TransformerDecoder
import logging
from d2l import torch as d2l

logging.propagate = False
logging.getLogger().setLevel(logging.ERROR)

import wandb


def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])
    net.apply(xavier_init_weights)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = d2l.MaskedSoftmaxCELoss()
    net.train()
    for epoch in range(num_epochs):
        timer = d2l.Timer()
        metric = d2l.Accumulator(2)  # Sum of training loss, no. of tokens
        for batch in data_iter:
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],
                               device=device).reshape(-1, 1)
            dec_input = d2l.concat([bos, Y[:, :-1]], 1)  # Teacher forcing
            Y_hat = net(X, dec_input, X_valid_len)
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()  # Make the loss scalar for `backward`
            d2l.grad_clipping(net, 1)
            num_tokens = Y_valid_len.sum()
            optimizer.step()
            with torch.no_grad():
                metric.add(l.sum(), num_tokens)
        wandb.log({"Train Loss": (metric[0] / metric[1])}, step=epoch)

    print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '
          f'tokens/sec on {str(device)}')

WANDB_PROJECT_NAME = 'transformer'
MODEL_FILE_NAME = 'transformer.h5'

wandb.init(project=WANDB_PROJECT_NAME)
wandb.watch_called = False

conf = wandb.config

conf.dropout = 0.1
conf.batch_size = 64
conf.num_steps = 10
conf.lr = 0.005
conf.epochs = 50
conf.no_cuda = False

conf.num_hiddens = 32
conf.num_layers = 2

conf.ffn_num_input = 32
conf.ffn_num_hiddens = 64
conf.num_heads = 4

conf.key_size = 32
conf.query_size = 32
conf.value_size = 32
conf.norm_shape = [32]

conf.seed = 42
conf.log_interval = 10

def main():
    use_cuda = not conf.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

    torch.manual_seed(conf.seed)
    numpy.random.seed(conf.seed)
    torch.backends.cudnn.deterministic = True

    train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(conf.batch_size, conf.num_steps)

    encoder = TransformerEncoder(len(src_vocab), conf.key_size, conf.query_size, conf.value_size, conf.num_hiddens,
                                 conf.norm_shape, conf.ffn_num_input, conf.ffn_num_hiddens, conf.num_heads,
                                 conf.num_layers, conf.dropout)
    decoder = TransformerDecoder(len(tgt_vocab), conf.key_size, conf.query_size, conf.value_size, conf.num_hiddens,
                                 conf.norm_shape, conf.ffn_num_input, conf.ffn_num_hiddens, conf.num_heads,
                                 conf.num_layers, conf.dropout)
    net = d2l.EncoderDecoder(encoder, decoder)

    wandb.watch(net, log="all")

    train_seq2seq(net, train_iter, conf.lr, conf.epochs, tgt_vocab, device)

    torch.save(net.state_dict(), MODEL_FILE_NAME)
    wandb.save(MODEL_FILE_NAME)
    engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
    fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
    for eng, fra in zip(engs, fras):
        translation, dec_attention_weight_seq = d2l.predict_seq2seq(
            net, eng, src_vocab, tgt_vocab, conf.num_steps, device, True)
        print(f'{eng} => {translation}, ',
            f'bleu {d2l.bleu(translation, fra, k=2):.3f}')

if __name__ == '__main__':
    main()
