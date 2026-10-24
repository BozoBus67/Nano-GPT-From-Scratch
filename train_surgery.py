import torch

from config import medium_gqa
from data_loading import load_corpus, train_val_split
from device import pick_device
from model import Transformer
from paths import CHECKPOINT_PATH, CORPUS_PATH
from train import MAX_LR, WEIGHT_DECAY, TrainingSession, train_loop


SOURCE_CHECKPOINT = CHECKPOINT_PATH.parent / '2026-10-18_surgery_source_m4.3M_s128.pt'
OUTPUT_CHECKPOINT = CHECKPOINT_PATH.parent / '2026-10-19_surgery_m4.3M_s128.pt'


def transfer_compatible_weights(new_model: Transformer, old_state_dict: dict) -> None:
  new_state = new_model.state_dict()
  transferred, reinit = [], []
  for key, new_tensor in new_state.items():
    if key in old_state_dict and old_state_dict[key].shape == new_tensor.shape:
      new_state[key] = old_state_dict[key].to(new_tensor.device)
      transferred.append(key)
    else:
      reinit.append(key)
  new_model.load_state_dict(new_state)
  print(f'=== SURGERY: transferred {len(transferred)} tensors, re-initialized {len(reinit)} ===', flush=True)
  print(f'  transferred: {transferred}', flush=True)
  print(f'  re-initialized: {reinit}', flush=True)


def main() -> None:
  print('=== SURGERY TRAINING START ===', flush=True)
  device = pick_device()
  old_ckpt = torch.load(SOURCE_CHECKPOINT, weights_only=False, map_location=device)
  tokenizer = old_ckpt['tokenizer']
  text = load_corpus(CORPUS_PATH)
  train_ids, val_ids = train_val_split(tokenizer.encode(text))
  config = medium_gqa(vocab_size=tokenizer.vocab_size)
  model = Transformer(config).to(device)
  transfer_compatible_weights(model, old_ckpt['model'])
  optimizer = torch.optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=WEIGHT_DECAY)
  session = TrainingSession(model=model, optimizer=optimizer, tokenizer=tokenizer, train_ids=train_ids, val_ids=val_ids, checkpoint_path=OUTPUT_CHECKPOINT)
  OUTPUT_CHECKPOINT.parent.mkdir(exist_ok=True)
  train_loop(session)


if __name__ == '__main__':
  main()
