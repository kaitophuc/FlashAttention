import math
import os
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

import time

class TokenEmbedding(nn.Module):
    def __init__ (self, vocab_size, d_model):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.scale = math.sqrt(d_model)

    # x: [batch_size, seq_len]: token indices
    def forward(self, x):
        emb = self.embedding(x) 
        return emb * self.scale
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout = 0.1, max_len = 512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        #positional encoding: [max_len, d_model]
        positional_encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
            #[ceil(d_model/2)]

        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        positional_encoding = positional_encoding.unsqueeze(0)
        self.register_buffer("positional_encoding", positional_encoding)

    def forward(self, x):
        #x: [batch_size, seq_len, d_model]
        seq_len = x.size(1)
        if (seq_len > self.positional_encoding.size(1)):
            raise ValueError(f"Sequence length {seq_len} exceeds maximum length {self.positional_encoding.size(1)}")
        pe = self.positional_encoding[:, :seq_len, :].to(device=x.device, dtype=x.dtype)
        return self.dropout(x + pe)

class InputEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, dropout = 0.1, max_len = 512):
        super(InputEmbedding, self).__init__()
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_len)

    def forward(self, x):
        #x: [batch_size, seq_len]: token indices
        token_emb = self.token_embedding(x) 
        return self.positional_encoding(token_emb)
    
class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout = 0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, Q, K, V, mask= None):
        #Q, K, V: [batch_size, num_heads, seq_len, d_k]
        #Output: [batch_size, num_heads, seq_len, d_k]
        d_k = Q.size(-1)
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        output = attn_weights @ V
        return output, attn_weights
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=8, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention(dropout)

    def split_heads(self, x):
        #x: [batch_size, seq_len, d_model]
        #Output: [batch_size, num_heads, seq_len, d_k]
        batch_size, seq_len, d_model = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)
    
    def combine_heads(self, x):
        #x: [batch_size, num_heads, seq_len, d_k]
        #Output: [batch_size, seq_len, d_model]
        batch_size, num_heads, seq_len, d_k = x.size()
        x = x.transpose(1, 2).contiguous().view(batch_size, seq_len, num_heads * d_k)
        return x

    def forward(self, Q, K, V, mask=None):
        #Q, K, V: [batch_size, seq_len, d_model]
        #Output: [batch_size, seq_len, d_model]
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
        attn_output, attn_weights = self.attention(Q, K, V, mask)
        attn_output = self.combine_heads(attn_output)
        output = self.W_o(attn_output)
        return output, attn_weights
    
class PositionWiseFeedForward(nn.Module):
    def __init__ (self, d_model = 512, d_ff = 2048, dropout = 0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        #x: [batch_size, seq_len, d_model]
        #Output: [batch_size, seq_len, d_model]
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class SublayerConnection(nn.Module):
    #Add and norm layer after each sublayer (self-attention, feed forward)
    def __init__(self, d_model = 512, dropout = 0.1):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer_output):
        #x: [batch_size, seq_len, d_model]
        #Output: [batch_size, seq_len, d_model]
        return self.norm(x + self.dropout(sublayer_output))
    
class EncoderLayer(nn.Module):
    def __init__ (self, d_model = 512, num_heads = 8, d_ff = 2048, dropout = 0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.sublayer1 = SublayerConnection(d_model, dropout)
        self.sublayer2 = SublayerConnection(d_model, dropout)

    def forward(self, x, mask=None):
        #x: [batch_size, seq_len, d_model]
        #Output: [batch_size, seq_len, d_model]
        attn_output, attn_weights = self.self_attn(x, x, x, mask)
        x = self.sublayer1(x, attn_output)
        ff_output = self.feed_forward(x)
        x = self.sublayer2(x, ff_output)
        return x, attn_weights
    
class DecoderLayer(nn.Module):
    def __init__ (self, d_model = 512, num_heads = 8, d_ff = 2048, dropout = 0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.sublayer1 = SublayerConnection(d_model, dropout)
        self.sublayer2 = SublayerConnection(d_model, dropout)
        self.sublayer3 = SublayerConnection(d_model, dropout)

    def forward(self, x, enc_output, self_mask=None, cross_mask=None):
        #x: [batch_size, seq_len, d_model]
        #enc_output: [batch_size, seq_len, d_model]
        #Output: [batch_size, seq_len, d_model]
        self_attn_output, self_attn_weights = self.self_attn(x, x, x, self_mask)
        x = self.sublayer1(x, self_attn_output)
        cross_attn_output, cross_attn_weights = self.cross_attn(x, enc_output, enc_output, cross_mask)
        x = self.sublayer2(x, cross_attn_output)
        ff_output = self.feed_forward(x)
        x = self.sublayer3(x, ff_output)
        return x, self_attn_weights, cross_attn_weights
    
class Encoder(nn.Module):
    def __init__ (self, num_layers = 6, d_model = 512, num_heads = 8, d_ff = 2048, dropout = 0.1):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        #x: [batch_size, seq_len, d_model]
        #Output: [batch_size, seq_len, d_model]
        attn_weights_list = []
        for layer in self.layers:
            x, attn_weights = layer(x, mask)
            attn_weights_list.append(attn_weights)
        return x, attn_weights_list
    
class Decoder(nn.Module):
    def __init__ (self, num_layers = 6, d_model = 512, num_heads = 8, d_ff = 2048, dropout = 0.1):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

    def forward(self, x, enc_output, self_mask=None, cross_mask=None):
        #x: [batch_size, seq_len, d_model]
        #enc_output: [batch_size, seq_len, d_model]
        #Output: [batch_size, seq_len, d_model]
        self_attn_weights_list = []
        cross_attn_weights_list = []
        for layer in self.layers:
            x, self_attn_weights, cross_attn_weights = layer(x, enc_output, self_mask, cross_mask)
            self_attn_weights_list.append(self_attn_weights)
            cross_attn_weights_list.append(cross_attn_weights)
        return x, self_attn_weights_list, cross_attn_weights_list
    
class Transformer(nn.Module):
    def __init__ (self, src_vocab_size, tgt_vocab_size, d_model = 512, num_heads = 8, d_ff = 2048, num_layers = 6, dropout = 0.1, max_len = 512):
        super(Transformer, self).__init__()
        self.src_embedding = InputEmbedding(src_vocab_size, d_model, dropout, max_len)
        self.tgt_embedding = InputEmbedding(tgt_vocab_size, d_model, dropout, max_len)
        self.encoder = Encoder(num_layers, d_model, num_heads, d_ff, dropout)
        self.decoder = Decoder(num_layers, d_model, num_heads, d_ff, dropout)
        self.output_linear = nn.Linear(d_model, tgt_vocab_size, bias=False)
        self.output_linear.weight = self.tgt_embedding.token_embedding.embedding.weight
        self.src_pad_idx = 0
        self.tgt_pad_idx = 0
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src, src_mask=None):
        #src: [batch_size, src_seq_len]: token indices
        #Output: [batch_size, src_seq_len, d_model]
        src_emb = self.src_embedding(src)
        enc_output, enc_attn_weights = self.encoder(src_emb, src_mask)
        return enc_output, enc_attn_weights
    
    def decode(self, tgt, enc_output, self_mask=None, cross_mask=None):
        #tgt: [batch_size, tgt_seq_len]: token indices
        #enc_output: [batch_size, src_seq_len, d_model]
        #Output: [batch_size, tgt_seq_len, d_model]
        tgt_emb = self.tgt_embedding(tgt)
        dec_output, self_attn_weights, cross_attn_weights = self.decoder(tgt_emb, enc_output, self_mask, cross_mask)
        return dec_output, self_attn_weights, cross_attn_weights
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, cross_mask=None):
        #src: [batch_size, src_seq_len]: token indices
        #tgt: [batch_size, tgt_seq_len]: token indices
        #Output: [batch_size, tgt_seq_len, tgt_vocab_size]
        src_mask = self.make_src_mask(src) if src_mask is None else src_mask
        tgt_mask = self.make_tgt_mask(tgt) if tgt_mask is None else tgt_mask
        cross_mask = src_mask if cross_mask is None else cross_mask
        enc_output, enc_attn_weights = self.encode(src, src_mask)
        dec_output, self_attn_weights, cross_attn_weights = self.decode(tgt, enc_output, tgt_mask, cross_mask)
        output = self.output_linear(dec_output)
        return output, enc_attn_weights, self_attn_weights, cross_attn_weights
    
    def make_src_mask(self, src):
        #src: [batch_size, src_seq_len]
        #Output: [batch_size, src_seq_len]
        src_mask = (src != self.src_pad_idx)
        return src_mask

    def make_tgt_padding_mask(self, tgt):
        #tgt: [batch_size, tgt_seq_len]
        #Output: [batch_size, tgt_seq_len]
        tgt_mask = (tgt != self.tgt_pad_idx)
        return tgt_mask
    
    def make_causal_mask(self, tgt):
        #tgt: [batch_size, tgt_seq_len]
        #Output: [tgt_seq_len, tgt_seq_len]
        T = tgt.size(1)
        causal_mask = torch.tril(torch.ones((T, T), device=tgt.device, dtype=torch.bool))
        return causal_mask
    
    def make_tgt_mask(self, tgt):
        #tgt: [batch_size, tgt_seq_len]
        #Output: [batch_size, tgt_seq_len, tgt_seq_len]
        tgt_pad_mask = self.make_tgt_padding_mask(tgt).unsqueeze(1)
        causal_mask = self.make_causal_mask(tgt).unsqueeze(0)
        tgt_mask = tgt_pad_mask & causal_mask
        return tgt_mask