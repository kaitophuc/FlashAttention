import argparse
import csv
import os
import re
from collections import Counter

import torch
from torch.utils.data import DataLoader, TensorDataset

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
    p.add_argument("--num-epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)

    p.add_argument("--max-train-rows", type=int, default=200000)
    p.add_argument("--max-eval-rows", type=int, default=5000)

    p.add_argument("--max-src-len", type=int, default=128)
    p.add_argument("--max-tgt-len", type=int, default=128)

    p.add_argument("--src-vocab-size", type=int, default=32000)
    p.add_argument("--tgt-vocab-size", type=int, default=32000)
    p.add_argument("--min-freq", type=int, default=2)

    p.add_argument("--lowercase", action="store_true")

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
            src = row[src_col].strip()
            tgt = row[tgt_col].strip()
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


def make_loader(src_tensor, tgt_tensor, batch_size, shuffle, num_workers, pin_memory):
    ds = TensorDataset(src_tensor, tgt_tensor)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


def main():
    args = parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)

    print("Reading CSV...")
    train_pairs = read_pairs(args.train_csv, args.src_col, args.tgt_col, args.max_train_rows)
    eval_pairs = read_pairs(args.eval_csv, args.src_col, args.tgt_col, args.max_eval_rows)
    print(f"Train pairs: {len(train_pairs)}")
    print(f"Eval pairs: {len(eval_pairs)}")

    if len(train_pairs) == 0 or len(eval_pairs) == 0:
        raise RuntimeError("No usable sentence pairs found.")

    print("Building vocab...")
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

    print(f"Final src vocab size: {len(src_stoi)}")
    print(f"Final tgt vocab size: {len(tgt_stoi)}")

    bos_idx = tgt_stoi[BOS_TOKEN]
    eos_idx = tgt_stoi[EOS_TOKEN]

    print("Tensorizing...")
    train_src, train_tgt = tensorize_pairs(
        train_pairs, src_stoi, tgt_stoi, args.max_src_len, args.max_tgt_len, args.lowercase
    )
    eval_src, eval_tgt = tensorize_pairs(
        eval_pairs, src_stoi, tgt_stoi, args.max_src_len, args.max_tgt_len, args.lowercase
    )

    pin = args.device.startswith("cuda") and torch.cuda.is_available()
    train_loader = make_loader(
        train_src, train_tgt, args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin
    )
    eval_loader = make_loader(
        eval_src, eval_tgt, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin
    )

    print("Training...")
    model, history = build_and_train_transformer(
        train_loader=train_loader,
        eval_loader=eval_loader,
        src_vocab_size=len(src_stoi),
        tgt_vocab_size=len(tgt_stoi),
        num_epochs=args.num_epochs,
        device=args.device,
        save_path=args.save_path,
        resume=args.resume,
        bos_idx=bos_idx,
        eos_idx=eos_idx,
        eval_bleu_loader=eval_loader,
    )

    print("Done.")
    if history["train_loss"]:
        print(f"Last train loss: {history['train_loss'][-1]:.4f}")
    if history["eval_loss"]:
        print(f"Last eval loss: {history['eval_loss'][-1]:.4f}")
    if history["eval_ppl"]:
        print(f"Last eval ppl: {history['eval_ppl'][-1]:.4f}")
    if history["eval_bleu"]:
        print(f"Last eval BLEU: {history['eval_bleu'][-1]:.4f}")


if __name__ == "__main__":
    main()
