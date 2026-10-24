import time

import torch

from device import pick_device
from model import Transformer
from paths import CHECKPOINT_PATH, PREPARED_DATA_PATH
from train import MAX_LR, WEIGHT_DECAY, TrainingSession, train_loop


def main() -> None:
  print('=== RESUME PRETRAIN ===', flush=True)
  device = pick_device()
  ckpt = torch.load(CHECKPOINT_PATH, weights_only=False, map_location=device)
  prepared = torch.load(PREPARED_DATA_PATH, weights_only=False)
  tokenizer = prepared['tokenizer']
  train_ids = prepared['train_ids']
  val_ids = prepared['val_ids']
  model = Transformer(ckpt['config']).to(device)
  model.load_state_dict(ckpt['model'])
  optimizer = torch.optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=WEIGHT_DECAY)
  optimizer.load_state_dict(ckpt['optimizer'])
  session = TrainingSession(model=model, optimizer=optimizer, tokenizer=tokenizer, train_ids=train_ids, val_ids=val_ids, checkpoint_path=CHECKPOINT_PATH, start_time=time.time(), prior_seconds=ckpt.get('cumulative_seconds', 0.0))
  print(f'=== resuming from step {ckpt["step"]}, cum {session.prior_seconds / 3600:.2f}h ===', flush=True)
  train_loop(session, start_step=ckpt['step'])


if __name__ == '__main__':
  main()
