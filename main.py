import argparse
import csv
import functools
import os
import re
from collections import Counter

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from transformer import build_and_train_transformer


SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]
PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN = SPECIALS


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--train-csv", type=str, required=True)
    p.add_argument("--eval-csv", type=str, required=True)

    p.add_argument("--src-col", type=str, default="fr")
    p.add_argument("--tgt-col", type=str, default="en")

    p.add_argument("--save-path", type=str, default="checkpoints/fr_en_baseline.pt")
    p.add_argument("--resume", action="store_true")

    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--distributed", action="store_true")
    p.add_argument("--backend", type=str, default="nccl")
    p.add_argument("--precision", type=str, default="amp", choices=["amp", "fp32"])
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--log-interval", type=int, default=20)
    p.add_argument("--num-epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--prefetch-factor", type=int, default=4)
    p.add_argument("--persistent-workers", dest="persistent_workers", action="store_true")
    p.add_argument("--no-persistent-workers", dest="persistent_workers", action="store_false")
    p.add_argument("--drop-last", dest="drop_last", action="store_true")
    p.add_argument("--no-drop-last", dest="drop_last", action="store_false")
    p.add_argument("--eval-every-epochs", type=int, default=1)
    p.add_argument("--eval-max-batches", type=int, default=None)
    p.add_argument("--disable-bleu", action="store_true")
    p.add_argument("--enable-bleu", dest="disable_bleu", action="store_false")
    p.add_argument("--bleu-only-final", action="store_true")
    p.add_argument("--bleu-decode-method", type=str, default="beam", choices=["beam", "greedy"])
    p.add_argument("--benchmark-throughput", action="store_true")
    p.add_argument("--baseline-tokens-per-sec", type=float, default=None)

    p.add_argument("--max-train-rows", type=int, default=200000)
    p.add_argument("--max-eval-rows", type=int, default=5000)

    p.add_argument("--max-src-len", type=int, default=128)
    p.add_argument("--max-tgt-len", type=int, default=128)

    p.add_argument("--src-vocab-size", type=int, default=32000)
    p.add_argument("--tgt-vocab-size", type=int, default=32000)
    p.add_argument("--min-freq", type=int, default=2)

    p.add_argument("--lowercase", action="store_true")

    p.set_defaults(persistent_workers=True)
    p.set_defaults(drop_last=True)
    p.set_defaults(disable_bleu=True)

    return p.parse_args()


def tokenize(text, lowercase=False):
    if lowercase:
        text = text.lower()
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


def read_pairs(csv_path, src_col, tgt_col, max_rows=None):
    pairs = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if src_col not in reader.fieldnames or tgt_col not in reader.fieldnames:
            raise ValueError(
                f"{csv_path} columns are {reader.fieldnames}, need src_col={src_col}, tgt_col={tgt_col}"
            )
        for i, row in enumerate(reader):
            src = row.get(src_col)
            tgt = row.get(tgt_col)
            if src is None or tgt is None:
                continue
            src = src.strip()
            tgt = tgt.strip()
            if not src or not tgt:
                continue
            if src and tgt:
                pairs.append((src, tgt))
            if max_rows is not None and len(pairs) >= max_rows:
                break
    return pairs


def build_vocab(texts, vocab_size, min_freq, lowercase=False):
    counter = Counter()
    for t in texts:
        counter.update(tokenize(t, lowercase=lowercase))

    stoi = {tok: i for i, tok in enumerate(SPECIALS)}
    itos = list(SPECIALS)

    max_new = max(0, vocab_size - len(SPECIALS))
    for tok, freq in counter.most_common():
        if freq < min_freq:
            continue
        if tok in stoi:
            continue
        stoi[tok] = len(itos)
        itos.append(tok)
        if len(itos) >= len(SPECIALS) + max_new:
            break

    return stoi, itos


def encode_sentence(text, stoi, max_len, add_bos_eos=False, lowercase=False):
    pad_idx = stoi[PAD_TOKEN]
    unk_idx = stoi[UNK_TOKEN]
    bos_idx = stoi[BOS_TOKEN]
    eos_idx = stoi[EOS_TOKEN]

    ids = [stoi.get(tok, unk_idx) for tok in tokenize(text, lowercase=lowercase)]

    if add_bos_eos:
        ids = [bos_idx] + ids[: max_len - 2] + [eos_idx]
    else:
        ids = ids[:max_len]

    if len(ids) < max_len:
        ids = ids + [pad_idx] * (max_len - len(ids))

    return ids


def tensorize_pairs(pairs, src_stoi, tgt_stoi, max_src_len, max_tgt_len, lowercase=False):
    src_ids = []
    tgt_ids = []

    for src_text, tgt_text in pairs:
        src_vec = encode_sentence(
            src_text, src_stoi, max_src_len, add_bos_eos=False, lowercase=lowercase
        )
        tgt_vec = encode_sentence(
            tgt_text, tgt_stoi, max_tgt_len, add_bos_eos=True, lowercase=lowercase
        )
        src_ids.append(src_vec)
        tgt_ids.append(tgt_vec)

    src_tensor = torch.tensor(src_ids, dtype=torch.long)
    tgt_tensor = torch.tensor(tgt_ids, dtype=torch.long)
    return src_tensor, tgt_tensor


def non_pad_length(sequence, pad_idx):
    non_pad = (sequence != pad_idx).nonzero(as_tuple=False)
    if non_pad.numel() == 0:
        return 1
    return int(non_pad[-1].item()) + 1


def trim_and_pad_collate(batch, pad_idx=0):
    src_batch, tgt_batch = zip(*batch)
    src = torch.stack(src_batch, dim=0)
    tgt = torch.stack(tgt_batch, dim=0)

    max_src_len = max(non_pad_length(row, pad_idx) for row in src)
    max_tgt_len = max(non_pad_length(row, pad_idx) for row in tgt)
    max_tgt_len = max(max_tgt_len, 2)

    src = src[:, :max_src_len].contiguous()
    tgt = tgt[:, :max_tgt_len].contiguous()
    return src, tgt


def make_loader(
    dataset,
    batch_size,
    shuffle,
    num_workers,
    pin_memory,
    drop_last,
    prefetch_factor,
    persistent_workers,
    sampler=None,
    collate_fn=None,
):
    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn,
    )
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
        kwargs["persistent_workers"] = persistent_workers
    return DataLoader(**kwargs)


def init_distributed(args):
    if not args.distributed:
        return {
            "enabled": False,
            "rank": 0,
            "local_rank": 0,
            "world_size": 1,
            "is_main": True,
        }

    required = ("RANK", "LOCAL_RANK", "WORLD_SIZE")
    missing = [k for k in required if k not in os.environ]
    if missing:
        raise RuntimeError(
            f"Distributed mode needs env vars {required}; missing {missing}. Use torchrun."
        )

    if not torch.cuda.is_available():
        raise RuntimeError("Distributed CUDA training requested but CUDA is unavailable.")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    try:
        dist.init_process_group(
            backend=args.backend,
            init_method="env://",
            device_id=local_rank,
        )
    except TypeError:
        # Backward compatibility for PyTorch versions without `device_id`.
        dist.init_process_group(backend=args.backend, init_method="env://")

    return {
        "enabled": True,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "is_main": rank == 0,
    }


def rank_print(dist_ctx, message):
    if dist_ctx["is_main"]:
        print(message)


def steady_value(values):
    if not values:
        return None
    if len(values) <= 1:
        return values[-1]
    steady = values[1:]
    return sum(steady) / len(steady)


def emit_benchmark_report(args, dist_ctx, history):
    if not args.benchmark_throughput:
        return
    tokens_per_sec = history.get("train_tokens_per_sec", [])
    steady_tokens_per_sec = steady_value(tokens_per_sec)
    if steady_tokens_per_sec is None:
        rank_print(dist_ctx, "Throughput benchmark: no train throughput metrics were recorded.")
        return

    device_count = dist_ctx["world_size"] if dist_ctx["enabled"] else 1
    rank_print(
        dist_ctx,
        f"Throughput benchmark (steady-state, excluding warmup epoch): "
        f"{steady_tokens_per_sec:.2f} tokens/sec on {device_count} GPU(s).",
    )

    if args.baseline_tokens_per_sec is not None and args.baseline_tokens_per_sec > 0:
        speedup = steady_tokens_per_sec / args.baseline_tokens_per_sec
        verdict = "PASS" if speedup >= 1.5 else "FAIL"
        rank_print(
            dist_ctx,
            f"Normalized speedup vs baseline: {speedup:.3f}x (target >= 1.5x) -> {verdict}",
        )


def main():
    args = parse_args()
    dist_ctx = None
    if args.eval_every_epochs < 1:
        raise ValueError("--eval-every-epochs must be >= 1")
    if args.grad_accum_steps < 1:
        raise ValueError("--grad-accum-steps must be >= 1")
    if args.prefetch_factor < 1:
        raise ValueError("--prefetch-factor must be >= 1")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    try:
        dist_ctx = init_distributed(args)
        if dist_ctx["enabled"]:
            device = f"cuda:{dist_ctx['local_rank']}"
        else:
            device = args.device

        if dist_ctx["is_main"]:
            os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)

        rank_print(dist_ctx, "Reading CSV...")
        train_pairs = read_pairs(args.train_csv, args.src_col, args.tgt_col, args.max_train_rows)
        eval_pairs = read_pairs(args.eval_csv, args.src_col, args.tgt_col, args.max_eval_rows)
        rank_print(dist_ctx, f"Train pairs: {len(train_pairs)}")
        rank_print(dist_ctx, f"Eval pairs: {len(eval_pairs)}")

        if len(train_pairs) == 0 or len(eval_pairs) == 0:
            raise RuntimeError("No usable sentence pairs found.")

        rank_print(dist_ctx, "Building vocab...")
        src_stoi, _ = build_vocab(
            [s for s, _ in train_pairs],
            vocab_size=args.src_vocab_size,
            min_freq=args.min_freq,
            lowercase=args.lowercase,
        )
        tgt_stoi, _ = build_vocab(
            [t for _, t in train_pairs],
            vocab_size=args.tgt_vocab_size,
            min_freq=args.min_freq,
            lowercase=args.lowercase,
        )

        rank_print(dist_ctx, f"Final src vocab size: {len(src_stoi)}")
        rank_print(dist_ctx, f"Final tgt vocab size: {len(tgt_stoi)}")

        bos_idx = tgt_stoi[BOS_TOKEN]
        eos_idx = tgt_stoi[EOS_TOKEN]

        rank_print(dist_ctx, "Tensorizing...")
        train_src, train_tgt = tensorize_pairs(
            train_pairs, src_stoi, tgt_stoi, args.max_src_len, args.max_tgt_len, args.lowercase
        )
        eval_src, eval_tgt = tensorize_pairs(
            eval_pairs, src_stoi, tgt_stoi, args.max_src_len, args.max_tgt_len, args.lowercase
        )

        train_ds = TensorDataset(train_src, train_tgt)
        eval_ds = TensorDataset(eval_src, eval_tgt)

        pin = device.startswith("cuda") and torch.cuda.is_available()
        collate_fn = functools.partial(trim_and_pad_collate, pad_idx=0)

        train_sampler = None
        eval_sampler = None
        if dist_ctx["enabled"]:
            train_sampler = DistributedSampler(
                train_ds,
                num_replicas=dist_ctx["world_size"],
                rank=dist_ctx["rank"],
                shuffle=True,
                drop_last=args.drop_last,
            )
            eval_sampler = DistributedSampler(
                eval_ds,
                num_replicas=dist_ctx["world_size"],
                rank=dist_ctx["rank"],
                shuffle=False,
                drop_last=False,
            )

        train_loader = make_loader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=pin,
            drop_last=args.drop_last,
            prefetch_factor=args.prefetch_factor,
            persistent_workers=args.persistent_workers,
            sampler=train_sampler,
            collate_fn=collate_fn,
        )
        eval_loader = make_loader(
            eval_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin,
            drop_last=False,
            prefetch_factor=args.prefetch_factor,
            persistent_workers=args.persistent_workers,
            sampler=eval_sampler,
            collate_fn=collate_fn,
        )

        eval_bleu_loader = None
        if not args.disable_bleu and dist_ctx["is_main"]:
            eval_bleu_loader = make_loader(
                eval_ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=pin,
                drop_last=False,
                prefetch_factor=args.prefetch_factor,
                persistent_workers=args.persistent_workers,
                sampler=None,
                collate_fn=collate_fn,
            )

        rank_print(
            dist_ctx,
            f"Training with device={device}, distributed={dist_ctx['enabled']}, "
            f"world_size={dist_ctx['world_size']}, precision={args.precision}",
        )
        _, history = build_and_train_transformer(
            train_loader=train_loader,
            eval_loader=eval_loader,
            src_vocab_size=len(src_stoi),
            tgt_vocab_size=len(tgt_stoi),
            num_epochs=args.num_epochs,
            device=device,
            save_path=args.save_path,
            resume=args.resume,
            bos_idx=bos_idx,
            eos_idx=eos_idx,
            eval_bleu_loader=eval_bleu_loader,
            eval_bleu_decode_method=args.bleu_decode_method,
            bleu_only_final=args.bleu_only_final,
            eval_every_epochs=args.eval_every_epochs,
            precision=args.precision,
            grad_accum_steps=args.grad_accum_steps,
            log_interval=args.log_interval,
            train_sampler=train_sampler,
            eval_max_batches=args.eval_max_batches,
            benchmark_throughput=args.benchmark_throughput,
            distributed=dist_ctx["enabled"],
            local_rank=dist_ctx["local_rank"],
            rank=dist_ctx["rank"],
            world_size=dist_ctx["world_size"],
        )

        rank_print(dist_ctx, "Done.")
        if history["train_loss"]:
            rank_print(dist_ctx, f"Last train loss: {history['train_loss'][-1]:.4f}")
        if history["eval_loss"]:
            rank_print(dist_ctx, f"Last eval loss: {history['eval_loss'][-1]:.4f}")
        if history["eval_ppl"]:
            rank_print(dist_ctx, f"Last eval ppl: {history['eval_ppl'][-1]:.4f}")
        if history["eval_bleu"]:
            rank_print(dist_ctx, f"Last eval BLEU: {history['eval_bleu'][-1]:.4f}")

        emit_benchmark_report(args, dist_ctx, history)
    finally:
        if dist_ctx is not None and dist_ctx["enabled"] and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
