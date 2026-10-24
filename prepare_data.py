import time
from pathlib import Path

import torch

from data_loading import iter_file_chunks, train_val_split
from paths import CORPUS_PATH_FULL, PREPARED_DATA_PATH
from tokenizer import BPETokenizer


VOCAB_SIZE = 512


def main() -> None:
  prepare(CORPUS_PATH_FULL, PREPARED_DATA_PATH, VOCAB_SIZE)


def prepare(corpus_path: Path, out_path: Path, vocab_size: int) -> None:
  print(f'=== PREPARE: {corpus_path} -> {out_path} ===', flush=True)
  t0 = time.perf_counter()
  tokenizer = BPETokenizer(iter_file_chunks(corpus_path), vocab_size)
  print(f'  tokenizer: vocab {tokenizer.vocab_size}, {len(tokenizer.merges)} merges, {time.perf_counter() - t0:.1f}s', flush=True)
  t1 = time.perf_counter()
  ids = encode_corpus(tokenizer, corpus_path)
  print(f'  encoded: {len(ids):,} tokens ({ids.element_size() * len(ids) / 1024 / 1024:.1f} MB), {time.perf_counter() - t1:.1f}s', flush=True)
  train_ids, val_ids = train_val_split(ids)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  torch.save({'tokenizer': tokenizer, 'train_ids': train_ids, 'val_ids': val_ids}, out_path)
  print(f'  saved. train {len(train_ids):,} / val {len(val_ids):,}. total {time.perf_counter() - t0:.1f}s', flush=True)


def encode_corpus(tokenizer: BPETokenizer, corpus_path: Path) -> torch.Tensor:
  pieces: list[torch.Tensor] = []
  for chunk in iter_file_chunks(corpus_path):
    chunk_ids = tokenizer.encode(chunk)
    pieces.append(torch.tensor(chunk_ids, dtype=torch.int16))
  return torch.cat(pieces)


if __name__ == '__main__':
  main()
