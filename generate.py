import torch

from device import pick_device
from model import Transformer
from paths import CHECKPOINT_PATH
from sampling import sample
from tokenizer import BPETokenizer, ENDOFTEXT


def _sample_ids(model: Transformer, prompt_ids: list[int], max_new_tokens: int, temperature: float, eot_id: int | None) -> list[int]:
  device = next(model.parameters()).device
  ids = list(prompt_ids)
  model.enable_cache()
  with torch.no_grad():
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    for _ in range(max_new_tokens):
      logits = model(x)[0]
      next_id = sample(logits[-1], temperature)
      if next_id == eot_id:
        break
      ids.append(next_id)
      x = torch.tensor([next_id], dtype=torch.long, device=device).unsqueeze(0)
  model.clear_cache()
  return ids


def generate_example(model: Transformer, tokenizer: BPETokenizer, prompt: str, max_new_tokens: int, temperature: float = 1.0) -> str:
  ids = tokenizer.encode(prompt)
  assert len(ids) + max_new_tokens <= model.config.max_seq_len, f'prompt ({len(ids)}) + max_new ({max_new_tokens}) exceeds max_seq_len ({model.config.max_seq_len})'
  eot_id = tokenizer.special_stoi.get(ENDOFTEXT)
  ids = _sample_ids(model, ids, max_new_tokens, temperature, eot_id)
  return tokenizer.decode(ids)


def main() -> None:
  device = pick_device()
  ckpt = torch.load(CHECKPOINT_PATH, weights_only=False, map_location=device)
  config = ckpt['config']
  tokenizer = ckpt['tokenizer']
  model = Transformer(config).to(device)
  model.load_state_dict(ckpt['model'])
  print(generate_example(model, tokenizer, 'Hello ', 100))


if __name__ == '__main__':
  main()
