import torch

from data_loading import load_corpus, train_val_split
from device import pick_device
from model import Transformer
from paths import CORPUS_PATH
from train import MAX_LR, WEIGHT_DECAY, TrainingSession, train_loop
from train_surgery import OUTPUT_CHECKPOINT


def main() -> None:
  print('=== RESUME SURGERY ===', flush=True)
  device = pick_device()
  ckpt = torch.load(OUTPUT_CHECKPOINT, weights_only=False, map_location=device)
  tokenizer = ckpt['tokenizer']
  text = load_corpus(CORPUS_PATH)
  train_ids, val_ids = train_val_split(tokenizer.encode(text))
  model = Transformer(ckpt['config']).to(device)
  model.load_state_dict(ckpt['model'])
  optimizer = torch.optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=WEIGHT_DECAY)
  optimizer.load_state_dict(ckpt['optimizer'])
  session = TrainingSession(model=model, optimizer=optimizer, tokenizer=tokenizer, train_ids=train_ids, val_ids=val_ids, checkpoint_path=OUTPUT_CHECKPOINT)
  print(f'=== resuming from step {ckpt["step"]} ===', flush=True)
  train_loop(session, start_step=ckpt['step'])


if __name__ == '__main__':
  main()
