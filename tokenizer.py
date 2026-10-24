import re
from collections import Counter
from typing import Iterable


ENDOFTEXT = '<|endoftext|>'
PRETOKEN_RE = re.compile(r'<\|endoftext\|>|\s*(?:(?!<\|endoftext\|>)\S)+|\s+')


class BPETokenizer:
  def __init__(self, chunks: str | Iterable[str], vocab_size: int) -> None:
    if isinstance(chunks, str):
      chunks = [chunks]
    word_counts, base_chars = self._pretokenize(chunks)
    base = sorted(base_chars)
    self.decoder: dict[int, str] = {i: c for i, c in enumerate(base)}
    self.base_stoi: dict[str, int] = {c: i for i, c in enumerate(base)}
    self.special_stoi: dict[str, int] = {}
    self._reserve_special(ENDOFTEXT)
    self.merges: dict[tuple[int, int], int] = {}
    id_counts = {tuple(self.base_stoi[c] for c in w): n for w, n in word_counts.items()}
    self._learn_merges(id_counts, vocab_size)
    self.vocab_size: int = len(self.decoder)

  @staticmethod
  def _pretokenize(chunks: Iterable[str]) -> tuple[Counter[str], set[str]]:
    word_counts: Counter[str] = Counter()
    base_chars: set[str] = set()
    for chunk in chunks:
      for m in PRETOKEN_RE.finditer(chunk):
        tok = m.group()
        base_chars.update(tok)
        if tok == ENDOFTEXT:
          continue
        word_counts[tok] += 1
    return word_counts, base_chars

  def _reserve_special(self, token: str) -> None:
    new_id = len(self.decoder)
    self.decoder[new_id] = token
    self.special_stoi[token] = new_id

  def _learn_merges(self, counts: dict[tuple[int, ...], int], vocab_size: int) -> None:
    while len(self.decoder) < vocab_size:
      pair_counts: Counter[tuple[int, int]] = Counter()
      for word, cnt in counts.items():
        for pair in zip(word, word[1:]):
          pair_counts[pair] += cnt
      if not pair_counts:
        break
      top_pair, top_count = pair_counts.most_common(1)[0]
      if top_count < 2:
        break
      new_id = len(self.decoder)
      self.decoder[new_id] = self.decoder[top_pair[0]] + self.decoder[top_pair[1]]
      self.merges[top_pair] = new_id
      counts = _merge_in_counts(counts, top_pair, new_id)

  def encode(self, text: str) -> list[int]:
    out: list[int] = []
    cache: dict[str, tuple[int, ...]] = {}
    for m in PRETOKEN_RE.finditer(text):
      tok = m.group()
      special_id = self.special_stoi.get(tok)
      if special_id is not None:
        out.append(special_id)
        continue
      cached = cache.get(tok)
      if cached is None:
        cached = tuple(self._encode_pretoken(tok))
        cache[tok] = cached
      out.extend(cached)
    return out

  def _encode_pretoken(self, pretoken: str) -> list[int]:
    ids = [self.base_stoi[c] for c in pretoken]
    while len(ids) >= 2:
      best_pos = -1
      best_id = -1
      for i, pair in enumerate(zip(ids, ids[1:])):
        merge_id = self.merges.get(pair)
        if merge_id is not None and (best_pos < 0 or merge_id < best_id):
          best_pos, best_id = i, merge_id
      if best_pos < 0:
        break
      ids = ids[:best_pos] + [best_id] + ids[best_pos + 2:]
    return ids

  def decode(self, ids: list[int]) -> str:
    return ''.join(self.decoder[i] for i in ids)


def _merge_in_counts(counts: dict[tuple[int, ...], int], pair: tuple[int, int], new_id: int) -> dict[tuple[int, ...], int]:
  out: dict[tuple[int, ...], int] = {}
  for word, cnt in counts.items():
    merged = _apply_merge(word, pair, new_id)
    out[merged] = out.get(merged, 0) + cnt
  return out


def _apply_merge(word: tuple[int, ...], pair: tuple[int, int], new_id: int) -> tuple[int, ...]:
  out: list[int] = []
  i = 0
  n = len(word)
  while i < n:
    if i < n - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
      out.append(new_id)
      i += 2
    else:
      out.append(word[i])
      i += 1
  return tuple(out)
