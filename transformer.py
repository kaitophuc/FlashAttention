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
        
class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size, padding_idx, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.smoothing_value = smoothing / (vocab_size - 2)
    
    def forward(self, pred, target):
        #pred: [batch_size, seq_len, vocab_size]
        #target: [batch_size, seq_len]
        pred = pred.reshape(-1, self.vocab_size)
        target = target.reshape(-1)
        log_probs = torch.log_softmax(pred, dim=-1)
        true_dist = torch.full_like(log_probs, self.smoothing_value)
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = (target != self.padding_idx)
        loss = - (true_dist * log_probs).sum(dim=-1)
        valid_loss = loss.masked_select(mask)
        return valid_loss.mean() if valid_loss.numel() > 0 else loss.new_tensor(0.0, device=pred.device)


def is_distributed_ready():
    return dist.is_available() and dist.is_initialized()


def get_world_size():
    return dist.get_world_size() if is_distributed_ready() else 1


def get_rank():
    return dist.get_rank() if is_distributed_ready() else 0


def is_main_process():
    return get_rank() == 0


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def all_reduce_sum(tensor):
    if is_distributed_ready():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def all_reduce_max(tensor):
    if is_distributed_ready():
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return tensor


def autocast_context(device, use_amp, amp_dtype):
    if not use_amp or device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=amp_dtype)
    
class NoamOpt:
    def __init__ (self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size

    def rate(self, step = None):
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))
    
    def step(self):
        self.update_learning_rate()
        self.optimizer.step()

    def update_learning_rate(self):
        self._step += 1
        lr = self.rate()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def zero_grad(self, set_to_none=True):
        self.optimizer.zero_grad(set_to_none=set_to_none)

def train_step(
    model,
    batch_src,
    batch_tgt,
    criterion,
    optimizer_wrapper,
    clip_grad=None,
    grad_accum_steps=1,
    do_optimizer_step=True,
    use_amp=False,
    amp_dtype=torch.bfloat16,
    scaler=None,
):
    base_model = unwrap_model(model)
    tgt_in = batch_tgt[:, :-1]
    tgt_out = batch_tgt[:, 1:]

    with autocast_context(batch_src.device, use_amp, amp_dtype):
        output, _, _, _ = model(batch_src, tgt_in)
        loss = criterion(output, tgt_out)

    scaled_loss = loss / grad_accum_steps
    if scaler is not None and scaler.is_enabled():
        scaler.scale(scaled_loss).backward()
    else:
        scaled_loss.backward()

    did_step = False
    if do_optimizer_step:
        if clip_grad is not None:
            if scaler is not None and scaler.is_enabled():
                scaler.unscale_(optimizer_wrapper.optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        if scaler is not None and scaler.is_enabled():
            optimizer_wrapper.update_learning_rate()
            scaler.step(optimizer_wrapper.optimizer)
            scaler.update()
        else:
            optimizer_wrapper.step()

        optimizer_wrapper.zero_grad(set_to_none=True)
        did_step = True

    non_pad = (tgt_out != base_model.tgt_pad_idx).sum().item()
    return loss.detach().item(), non_pad, did_step

@torch.no_grad()
def greedy_decode(model, src, max_len, bos_idx, eos_idx):
    model.eval()

    src_mask = model.make_src_mask(src)
    memory, _ = model.encode(src, src_mask)

    ys = torch.full((src.size(0), 1), bos_idx, dtype=torch.long, device=src.device)
    finished = torch.zeros(src.size(0), dtype=torch.bool, device=src.device)

    for _ in range(max_len - 1):
        tgt_mask = model.make_tgt_mask(ys)
        out, _, _ = model.decode(ys, memory, tgt_mask, src_mask)
        logits = model.output_linear(out[:, -1, :]) # last-step logits
        next_token = logits.argmax(dim=-1, keepdim=True)

        ys = torch.cat([ys, next_token], dim=1)
        finished = finished | (next_token.squeeze(1) == eos_idx)
        if finished.all():
            break
    return ys

@torch.no_grad()
def beam_search_decode(model, src, max_len, bos_idx, eos_idx, beam_size = 4, alpha = 0.6):
    model.eval()

    src_mask = model.make_src_mask(src)
    memory, _ = model.encode(src, src_mask)

    batch_size = src.size(0)
    best_sequences = []

    def length_penalty(length, alpha):
        return ((5 + length) / 6) ** alpha
    
    for i in range(batch_size):
        #Work with one example at a time
        memory_i = memory[i:i+1]
        src_mask_i = src_mask[i:i+1]

        beams = [(torch.full((1, 1), bos_idx, dtype=torch.long, device=src.device), 0.0, False)] # (sequence, score, finished)

        for _ in range(max_len - 1):
            new_beams = []
            for seq, score, finished in beams:
                if finished:
                    new_beams.append((seq, score, True))
                    continue

                tgt_mask = model.make_tgt_mask(seq)
                out, _, _ = model.decode(seq, memory_i, tgt_mask, src_mask_i)
                logits = model.output_linear(out[:, -1, :]) # last-step logits
                log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)

                topk_log_probs, topk_indices = log_probs.topk(beam_size)

                for k in range(topk_indices.size(0)):
                    token_idx = topk_indices[k].view(1, 1)
                    new_seq = torch.cat([seq, token_idx], dim=1)
                    new_log_prob = score + topk_log_probs[k].item()
                    is_finished = (token_idx.item() == eos_idx)
                    new_beams.append((new_seq, new_log_prob, is_finished))

            new_beams.sort(key=lambda x: x[1] / length_penalty(x[0].size(1), alpha), reverse=True)
            beams = new_beams[:beam_size]

            if all(finished for _, _, finished in beams):
                break

        best_seq, _ , _ = max(beams, key=lambda x: x[1] / length_penalty(x[0].size(1), alpha))
        best_sequences.append(best_seq)

    max_len = max(seq.size(1) for seq in best_sequences)
    padded_sequences = torch.full((batch_size, max_len), eos_idx, dtype=torch.long, device=src.device)
    for i, seq in enumerate(best_sequences):
        padded_sequences[i, :seq.size(1)] = seq.squeeze(0)
    return padded_sequences

@torch.no_grad()
def eval_step(model, batch_src, batch_tgt):
    model.eval()
    base_model = unwrap_model(model)

    tgt_in = batch_tgt[:, :-1]
    tgt_out = batch_tgt[:, 1:]

    output, _, _, _ = model(batch_src, tgt_in)

    log_probs = torch.log_softmax(output, dim=-1)
    nll = -log_probs.gather(dim=-1, index=tgt_out.unsqueeze(-1)).squeeze(-1)
    mask = (tgt_out != base_model.tgt_pad_idx)
    nll = nll.masked_select(mask)

    loss = nll.mean() if nll.numel() > 0 else nll.new_tensor(0.0, device=output.device)
    non_pad = (tgt_out != base_model.tgt_pad_idx).sum().item()
    ppl = math.exp(loss.item()) if loss.item() < 20 else float('inf')
    return loss.item(), ppl, non_pad

def run_train_epoch(
    model,
    train_loader,
    criterion,
    optimizer_wrapper,
    clip_grad=None,
    grad_accum_steps=1,
    use_amp=False,
    amp_dtype=torch.bfloat16,
    scaler=None,
    log_interval=20,
):
    model.train()
    total_loss = 0.0
    total_tokens = 0
    total_data_time = 0.0
    total_step_time = 0.0
    local_optimizer_steps = 0

    device = next(model.parameters()).device
    num_batches = len(train_loader)
    if num_batches == 0:
        return 0.0, {
            "tokens_per_sec": 0.0,
            "steps_per_sec": 0.0,
            "avg_data_time": 0.0,
            "avg_step_time": 0.0,
            "gpu_mem_gb": 0.0,
            "optimizer_steps": 0,
        }

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    epoch_start_time = time.time()
    optimizer_wrapper.zero_grad(set_to_none=True)
    last_step_end = time.time()

    for step, (batch_src, batch_tgt) in enumerate(train_loader, start=1):
        data_time = time.time() - last_step_end
        step_start = time.time()

        batch_src, batch_tgt = move_batch_to_device(batch_src, batch_tgt, device)
        should_step = (step % grad_accum_steps == 0) or (step == num_batches)
        loss, non_pad, did_step = train_step(
            model,
            batch_src,
            batch_tgt,
            criterion,
            optimizer_wrapper,
            clip_grad=clip_grad,
            grad_accum_steps=grad_accum_steps,
            do_optimizer_step=should_step,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            scaler=scaler,
        )
        if did_step:
            local_optimizer_steps += 1

        total_loss += loss * non_pad
        total_tokens += non_pad
        total_data_time += data_time
        total_step_time += time.time() - step_start

        elapsed = time.time() - epoch_start_time
        avg_step = elapsed / step if step > 0 else 0
        eta_seconds = avg_step * (num_batches - step)

        if is_main_process() and (step % log_interval == 0 or step == num_batches):
            print(
                f"[train] step {step}/{num_batches} "
                f"elapsed={elapsed/60:.1f}m eta={eta_seconds/60:.1f}m",
                end="\r",
                flush=True,
            )
        last_step_end = time.time()

    if is_main_process():
        print()

    epoch_wall = time.time() - epoch_start_time
    gpu_mem_gb = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        if device.type == "cuda"
        else 0.0
    )

    sum_stats = torch.tensor(
        [
            total_loss,
            float(total_tokens),
            total_data_time,
            total_step_time,
            float(num_batches),
            float(local_optimizer_steps),
        ],
        dtype=torch.float64,
        device=device,
    )
    max_stats = torch.tensor(
        [epoch_wall, gpu_mem_gb],
        dtype=torch.float64,
        device=device,
    )
    all_reduce_sum(sum_stats)
    all_reduce_max(max_stats)

    global_total_loss = sum_stats[0].item()
    global_total_tokens = int(sum_stats[1].item())
    global_total_data_time = sum_stats[2].item()
    global_total_step_time = sum_stats[3].item()
    global_total_batches = max(1.0, sum_stats[4].item())
    global_optimizer_steps = max(1.0, sum_stats[5].item())
    global_epoch_wall = max_stats[0].item()
    global_gpu_mem_gb = max_stats[1].item()

    avg_loss = global_total_loss / global_total_tokens if global_total_tokens > 0 else 0.0
    stats = {
        "tokens_per_sec": (global_total_tokens / global_epoch_wall) if global_epoch_wall > 0 else 0.0,
        "steps_per_sec": (global_optimizer_steps / global_epoch_wall) if global_epoch_wall > 0 else 0.0,
        "avg_data_time": global_total_data_time / global_total_batches,
        "avg_step_time": global_total_step_time / global_total_batches,
        "gpu_mem_gb": global_gpu_mem_gb,
        "optimizer_steps": int(global_optimizer_steps),
    }
    return avg_loss, stats


def run_eval_epoch(model, eval_loader, max_batches=None):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    seen_batches = 0

    device = next(model.parameters()).device
    eval_start_time = time.time()

    for batch_src, batch_tgt in eval_loader:
        if max_batches is not None and seen_batches >= max_batches:
            break
        batch_src, batch_tgt = move_batch_to_device(batch_src, batch_tgt, device)
        loss, _, non_pad = eval_step(model, batch_src, batch_tgt)
        total_loss += loss * non_pad
        total_tokens += non_pad
        seen_batches += 1

    eval_wall = time.time() - eval_start_time
    sum_stats = torch.tensor(
        [total_loss, float(total_tokens)],
        dtype=torch.float64,
        device=device,
    )
    max_stats = torch.tensor([eval_wall], dtype=torch.float64, device=device)
    all_reduce_sum(sum_stats)
    all_reduce_max(max_stats)

    global_total_loss = sum_stats[0].item()
    global_total_tokens = int(sum_stats[1].item())
    avg_loss = global_total_loss / global_total_tokens if global_total_tokens > 0 else 0.0
    ppl = math.exp(avg_loss) if avg_loss < 20 else float('inf')
    stats = {"tokens": global_total_tokens, "wall_time": max_stats[0].item()}
    return avg_loss, ppl, stats

def save_checkpoint(model, optimizer_wrapper, epoch, path, best_eval_loss):
    base_model = unwrap_model(model)
    checkpoint = {
        'model_state_dict': base_model.state_dict(),
        'optimizer_state_dict': optimizer_wrapper.optimizer.state_dict(),
        'epoch': epoch,
        'step': optimizer_wrapper._step,
        'best_eval_loss': best_eval_loss
    }
    torch.save(checkpoint, path)

def load_checkpoint(model, optimizer_wrapper, path):
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer_wrapper.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    device = next(model.parameters()).device
    for state in optimizer_wrapper.optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    optimizer_wrapper._step = checkpoint['step']
    start_epoch = checkpoint['epoch'] + 1
    return start_epoch

def fit(
    model,
    train_loader,
    eval_loader,
    criterion,
    optimizer_wrapper,
    num_epochs,
    clip_grad=None,
    save_path=None,
    start_epoch=0,
    bos_idx=None,
    eos_idx=None,
    eval_bleu_loader=None,
    eval_bleu_decode_method="beam",
    eval_bleu_beam_size=4,
    eval_bleu_alpha=0.6,
    eval_every_epochs=1,
    grad_accum_steps=1,
    precision="amp",
    log_interval=20,
    train_sampler=None,
    eval_max_batches=None,
    benchmark_throughput=False,
):
    best_eval_loss = float('inf')
    if save_path is not None and start_epoch > 0:
        ckpt = torch.load(save_path, map_location='cpu')
        best_eval_loss = ckpt.get('best_eval_loss', float('inf'))

    device = next(model.parameters()).device
    use_amp = (precision == "amp") and (device.type == "cuda")
    amp_dtype = torch.bfloat16
    if use_amp and not torch.cuda.is_bf16_supported():
        amp_dtype = torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))

    if is_main_process():
        precision_name = "fp32"
        if use_amp:
            precision_name = "bf16" if amp_dtype == torch.bfloat16 else "fp16"
        print(f"[setup] precision={precision_name}, grad_accum_steps={grad_accum_steps}")

    history = {
        "train_loss": [],
        "eval_loss": [],
        "eval_ppl": [],
        "eval_bleu": [],
        "train_tokens_per_sec": [],
        "train_steps_per_sec": [],
        "train_avg_data_time": [],
        "train_avg_step_time": [],
        "train_gpu_mem_gb": [],
        "effective_global_batch": [],
    }

    for epoch in range(start_epoch, num_epochs):
        if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)

        train_loss, train_stats = run_train_epoch(
            model,
            train_loader,
            criterion,
            optimizer_wrapper,
            clip_grad=clip_grad,
            grad_accum_steps=grad_accum_steps,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            scaler=scaler,
            log_interval=log_interval,
        )

        should_eval = ((epoch + 1) % eval_every_epochs == 0) or (epoch == num_epochs - 1)

        effective_global_batch = train_loader.batch_size * get_world_size() * grad_accum_steps
        history["train_loss"].append(train_loss)
        history["train_tokens_per_sec"].append(train_stats["tokens_per_sec"])
        history["train_steps_per_sec"].append(train_stats["steps_per_sec"])
        history["train_avg_data_time"].append(train_stats["avg_data_time"])
        history["train_avg_step_time"].append(train_stats["avg_step_time"])
        history["train_gpu_mem_gb"].append(train_stats["gpu_mem_gb"])
        history["effective_global_batch"].append(effective_global_batch)

        if should_eval:
            eval_loss, eval_ppl, _ = run_eval_epoch(model, eval_loader, max_batches=eval_max_batches)
            bleu_score = None

            if eval_bleu_loader is not None and bos_idx is not None and eos_idx is not None and is_main_process():
                base_model = unwrap_model(model)
                bleu_score = evaluate_bleu(
                    base_model,
                    eval_bleu_loader,
                    bos_idx,
                    eos_idx,
                    base_model.tgt_pad_idx,
                    decode_method=eval_bleu_decode_method,
                    beam_size=eval_bleu_beam_size,
                    alpha=eval_bleu_alpha,
                )
                print(
                    f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - "
                    f"Eval Loss: {eval_loss:.4f} - Eval PPL: {eval_ppl:.4f} - "
                    f"Eval BLEU: {bleu_score:.4f} - Tokens/s: {train_stats['tokens_per_sec']:.1f} - "
                    f"Data: {train_stats['avg_data_time']*1000:.2f}ms - Step: {train_stats['avg_step_time']*1000:.2f}ms - "
                    f"GPU Mem: {train_stats['gpu_mem_gb']:.2f}GB - Effective Global Batch: {effective_global_batch}"
                )
                history["eval_bleu"].append(bleu_score)
            elif is_main_process():
                print(
                    f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - "
                    f"Eval Loss: {eval_loss:.4f} - Eval PPL: {eval_ppl:.4f} - "
                    f"Tokens/s: {train_stats['tokens_per_sec']:.1f} - "
                    f"Data: {train_stats['avg_data_time']*1000:.2f}ms - Step: {train_stats['avg_step_time']*1000:.2f}ms - "
                    f"GPU Mem: {train_stats['gpu_mem_gb']:.2f}GB - Effective Global Batch: {effective_global_batch}"
                )

            if is_distributed_ready():
                dist.barrier()

            if save_path is not None and eval_loss < best_eval_loss and is_main_process():
                best_eval_loss = eval_loss
                save_checkpoint(model, optimizer_wrapper, epoch, save_path, best_eval_loss)
                print(f"Saved checkpoint at epoch {epoch+1} with eval loss {eval_loss:.4f}")

            history["eval_loss"].append(eval_loss)
            history["eval_ppl"].append(eval_ppl)
        else:
            if is_main_process():
                print(
                    f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - "
                    f"Eval skipped (every {eval_every_epochs} epochs) - "
                    f"Tokens/s: {train_stats['tokens_per_sec']:.1f} - "
                    f"Data: {train_stats['avg_data_time']*1000:.2f}ms - Step: {train_stats['avg_step_time']*1000:.2f}ms - "
                    f"GPU Mem: {train_stats['gpu_mem_gb']:.2f}GB - Effective Global Batch: {effective_global_batch}"
                )

    if benchmark_throughput and is_main_process() and history["train_tokens_per_sec"]:
        steady = history["train_tokens_per_sec"][1:] if len(history["train_tokens_per_sec"]) > 1 else history["train_tokens_per_sec"]
        steady_tps = sum(steady) / len(steady)
        print(f"[benchmark] steady_state_tokens_per_sec={steady_tps:.2f}")
    return history

@torch.no_grad()
def translate_batch(model, src, bos_idx, eos_idx, max_len=128, method="beam", beam_size=4, alpha=0.6):
    model.eval()
    if method == "beam":
        return beam_search_decode(model, src, max_len, bos_idx, eos_idx, beam_size=beam_size, alpha=alpha)
    elif method == "greedy":
        return greedy_decode(model, src, max_len, bos_idx, eos_idx)
    else:
        raise ValueError(f"Unknown decode method: {method}")
    
@torch.no_grad()
def sanity_check_transformer(
    src_vocab_size=100,
    tgt_vocab_size=120,
    batch_size=2,
    src_len=10,
    tgt_len=12,
    d_model=512,
    device="cpu",
):
    model = Transformer(src_vocab_size, tgt_vocab_size, d_model=d_model).to(device)
    model.eval()

    src = torch.randint(1, src_vocab_size, (batch_size, src_len), device=device)
    tgt = torch.randint(1, tgt_vocab_size, (batch_size, tgt_len), device=device)

    logits, enc_attn, self_attn, cross_attn = model(src, tgt[:, :-1])

    assert logits.shape == (batch_size, tgt_len - 1, tgt_vocab_size)
    assert len(enc_attn) == 6
    assert len(self_attn) == 6
    assert len(cross_attn) == 6

    out_greedy = greedy_decode(model, src, max_len=16, bos_idx=1, eos_idx=2)
    out_beam = beam_search_decode(model, src, max_len=16, bos_idx=1, eos_idx=2, beam_size=4, alpha=0.6)

    assert out_greedy.dim() == 2 and out_greedy.size(0) == batch_size
    assert out_beam.dim() == 2 and out_beam.size(0) == batch_size

    return {
        "logits_shape": tuple(logits.shape),
        "greedy_shape": tuple(out_greedy.shape),
        "beam_shape": tuple(out_beam.shape),
    }

def move_batch_to_device(batch_src, batch_tgt, device):
    return batch_src.to(device, non_blocking=True), batch_tgt.to(device, non_blocking=True)

def set_seed(seed=42, deterministic=False):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def build_and_train_transformer(
    train_loader,
    eval_loader,
    src_vocab_size,
    tgt_vocab_size,
    num_epochs=10,
    device="cuda",
    save_path=None,
    resume=False,
    bos_idx=None,
    eos_idx=None,
    eval_bleu_loader=None,
    eval_bleu_decode_method="beam",
    eval_every_epochs=1,
    precision="amp",
    grad_accum_steps=1,
    log_interval=20,
    train_sampler=None,
    eval_max_batches=None,
    benchmark_throughput=False,
    distributed=False,
    local_rank=0,
    rank=0,
    world_size=1,
):
    set_seed(42 + rank)
    model = Transformer(src_vocab_size, tgt_vocab_size).to(device)

    criterion = LabelSmoothingLoss(tgt_vocab_size, padding_idx=model.tgt_pad_idx, smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9, lr=0.0)
    optimizer_wrapper = NoamOpt(model_size=512, factor=1, warmup=4000, optimizer=optimizer)

    if resume and save_path is not None:
        if not os.path.exists(save_path):
            if is_main_process():
                print(f"No checkpoint found at {save_path}, starting from scratch.")
            start_epoch = 0
        else:
            start_epoch = load_checkpoint(model, optimizer_wrapper, save_path)
            if is_main_process():
                print(f"Resuming training from epoch {start_epoch}")
    else:        
        start_epoch = 0

    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )

    history = fit(
        model,
        train_loader,
        eval_loader,
        criterion,
        optimizer_wrapper,
        num_epochs,
        save_path=save_path,
        start_epoch=start_epoch,
        bos_idx=bos_idx,
        eos_idx=eos_idx,
        eval_bleu_loader=eval_bleu_loader,
        eval_bleu_decode_method=eval_bleu_decode_method,
        eval_every_epochs=eval_every_epochs,
        grad_accum_steps=grad_accum_steps,
        precision=precision,
        log_interval=log_interval,
        train_sampler=train_sampler,
        eval_max_batches=eval_max_batches,
        benchmark_throughput=benchmark_throughput,
    )
    return model, history

@torch.no_grad()
def evaluate_bleu(model, eval_loader, bos_idx, eos_idx, pad_idx, decode_method="beam", beam_size=4, alpha=0.6):
    from collections import Counter

    def trim_tokens(tokens):
        out = []
        for t in tokens:
            if t == eos_idx:
                break
            if t == bos_idx or t == pad_idx:
                continue
            out.append(t)
        return out

    def ngrams(tokens, n):
        return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

    def corpus_bleu4(hypotheses, references):
        max_order = 4
        clipped_counts = [0] * max_order
        total_counts = [0] * max_order
        hyp_len = 0
        ref_len = 0

        for hyp, ref in zip(hypotheses, references):
            hyp_len += len(hyp)
            ref_len += len(ref)

            for n in range(1, max_order + 1):
                hyp_ngrams = Counter(ngrams(hyp, n))
                ref_ngrams = Counter(ngrams(ref, n))

                total_counts[n - 1] += sum(hyp_ngrams.values())
                for ng, c in hyp_ngrams.items():
                    clipped_counts[n - 1] += min(c, ref_ngrams.get(ng, 0))

        if hyp_len == 0:
            return 0.0

        precisions = []
        for i in range(max_order):
            if total_counts[i] == 0 or clipped_counts[i] == 0:
                return 0.0
            precisions.append(clipped_counts[i] / total_counts[i])

        bp = 1.0 if hyp_len > ref_len else math.exp(1.0 - (ref_len / hyp_len))
        bleu = bp * math.exp(sum(0.25 * math.log(p) for p in precisions))
        return float(bleu)

    model.eval()
    device = next(model.parameters()).device

    hypotheses = []
    references = []

    for batch_src, batch_tgt in eval_loader:
        batch_src, batch_tgt = move_batch_to_device(batch_src, batch_tgt, device)

        decoded = translate_batch(
            model,
            batch_src,
            bos_idx,
            eos_idx,
            method=decode_method,
            beam_size=beam_size,
            alpha=alpha,
        )

        for i in range(batch_src.size(0)):
            hyp = trim_tokens(decoded[i].tolist())
            ref = trim_tokens(batch_tgt[i].tolist())
            hypotheses.append(hyp)
            references.append(ref)

    return corpus_bleu4(hypotheses, references)

def average_checkpoints(model, checkpoint_paths, device="cpu"):
    if not checkpoint_paths:
        raise ValueError("checkpoint_paths must be a non-empty list of paths.")

    state_dicts = []
    for path in checkpoint_paths:
        checkpoint = torch.load(path, map_location=device)
        if "model_state_dict" not in checkpoint:
            raise KeyError(f"'model_state_dict' not found in checkpoint: {path}")
        state_dicts.append(checkpoint["model_state_dict"])

    ref_keys = set(state_dicts[0].keys())
    for i, sd in enumerate(state_dicts[1:], start=1):
        if set(sd.keys()) != ref_keys:
            raise ValueError(f"Checkpoint at index {i} has mismatched state_dict keys.")

    avg_state_dict = {}
    n = len(state_dicts)

    for key in state_dicts[0].keys():
        tensor0 = state_dicts[0][key]

        if torch.is_floating_point(tensor0):
            acc = tensor0.clone()
            for sd in state_dicts[1:]:
                acc = acc + sd[key]
            avg_state_dict[key] = acc / n
        else:
            # Keep non-floating tensors (e.g. integer buffers) unchanged
            avg_state_dict[key] = tensor0.clone()

    model.load_state_dict(avg_state_dict)
    return model

@torch.no_grad()
def evaluate_with_checkpoint_averaging(
    model,
    checkpoint_paths,
    eval_loader,
    bos_idx,
    eos_idx,
    pad_idx,
    device="cpu",
    decode_method="beam",
    beam_size=4,
    alpha=0.6,
):
    if not checkpoint_paths:
        raise ValueError("checkpoint_paths must be a non-empty list of paths.")

    average_checkpoints(model, checkpoint_paths, device=device)
    model = model.to(device)
    model.eval()

    bleu = evaluate_bleu(
        model=model,
        eval_loader=eval_loader,
        bos_idx=bos_idx,
        eos_idx=eos_idx,
        pad_idx=pad_idx,
        decode_method=decode_method,
        beam_size=beam_size,
        alpha=alpha,
    )
    return bleu

if __name__ == "__main__":
    # Example usage (replace these placeholders with real objects before running this file directly).
    raise RuntimeError(
        "Set up train_loader_small, eval_loader_small, src_vocab_size, tgt_vocab_size, "
        "bos_idx, and eos_idx before running transformer.py as a script."
    )
