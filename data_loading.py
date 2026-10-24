from pathlib import Path
from typing import Iterator

import torch


CHUNK_BYTES = 16 * 1024 * 1024  # read corpus in roughly 16MB line-batched chunks


def load_corpus(path: str | Path) -> str:
  with open(path) as f:
    return f.read()


def iter_file_chunks(path: str | Path, chunk_bytes: int = CHUNK_BYTES) -> Iterator[str]:
  with open(path) as f:
    buf: list[str] = []
    total = 0
    for line in f:
      buf.append(line)
      total += len(line)
      if total >= chunk_bytes:
        yield ''.join(buf)
        buf = []
        total = 0
    if buf:
      yield ''.join(buf)


def train_val_split(ids: torch.Tensor, val_fraction: float = 0.1) -> tuple[torch.Tensor, torch.Tensor]:
  n = int(len(ids) * (1 - val_fraction))
  return ids[:n], ids[n:]


def get_batch(ids: torch.Tensor, batch_size: int, block_size: int): # 1dtensor, int, int -> (2dtensor, 2dtensor)
  starts = torch.randint(0, len(ids) - block_size - 1, (batch_size,)).tolist()
  x = torch.stack([ids[s:s+block_size] for s in starts]).long()
  y = torch.stack([ids[s+1:s+block_size+1] for s in starts]).long()
  return x, y
