import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from config import large_gqa
from data_loading import get_batch
from device import pick_device
from generate import generate_example
from model import Transformer
from paths import CHECKPOINT_PATH, PREPARED_DATA_PATH
from tokenizer import BPETokenizer


BATCH_SIZE = 8
BLOCK_SIZE = 512
ACCUM_STEPS = 16
MAX_LR = 3e-4
MIN_LR = 3e-5
WARMUP_STEPS = 200
NUM_STEPS = 10000
LOG_EVERY = 100
SAVE_EVERY = 100
EVAL_EVERY = 500
VAL_BATCHES = 10
GRAD_CLIP = 1.0
WEIGHT_DECAY = 0.1
SAMPLE_PROMPT = 'Once upon a time there was a'
SAMPLE_MAX_NEW_TOKENS = 50
SAMPLES_PATH = CHECKPOINT_PATH.parent / 'samples.jsonl'


@dataclass
class TrainingSession:
  model: Transformer
  optimizer: torch.optim.AdamW
  tokenizer: BPETokenizer
  train_ids: torch.Tensor
  val_ids: torch.Tensor
  checkpoint_path: Path
  start_time: float
  prior_seconds: float


def get_lr(step: int) -> float:
  if step < WARMUP_STEPS:
    return MAX_LR * (step + 1) / WARMUP_STEPS
  if step >= NUM_STEPS:
    return MIN_LR
  progress = (step - WARMUP_STEPS) / (NUM_STEPS - WARMUP_STEPS)
  return MIN_LR + 0.5 * (MAX_LR - MIN_LR) * (1 + math.cos(math.pi * progress))


def main() -> None:
  prepared = torch.load(PREPARED_DATA_PATH, weights_only=False)
  tokenizer = prepared['tokenizer']
  train_ids = prepared['train_ids']
  val_ids = prepared['val_ids']
  config = large_gqa(vocab_size=tokenizer.vocab_size)
  model = Transformer(config).to(pick_device())
  optimizer = torch.optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=WEIGHT_DECAY)
  session = TrainingSession(model=model, optimizer=optimizer, tokenizer=tokenizer, train_ids=train_ids, val_ids=val_ids, checkpoint_path=CHECKPOINT_PATH, start_time=time.time(), prior_seconds=0.0)
  CHECKPOINT_PATH.parent.mkdir(exist_ok=True)
  train_loop(session)


def train_loop(session: TrainingSession, start_step: int = 0) -> None:
  for step in range(start_step, NUM_STEPS):
    lr = get_lr(step)
    session.optimizer.param_groups[0]['lr'] = lr
    total_loss = training_step(session)
    periodic_log_eval_and_save(session, step, total_loss, lr)
  val_loss = evaluate(session)
  print(f'step {NUM_STEPS}: val loss {val_loss:.4f}', flush=True)
  save_checkpoint(session, NUM_STEPS)


def training_step(session: TrainingSession) -> float:
  device = next(session.model.parameters()).device
  session.optimizer.zero_grad()
  total_loss = 0.0
  for _ in range(ACCUM_STEPS):
    x, y = get_batch(session.train_ids, BATCH_SIZE, BLOCK_SIZE)
    x, y = x.to(device), y.to(device)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
      logits = session.model(x)
      loss = F.cross_entropy(logits.reshape(-1, session.model.config.vocab_size), y.reshape(-1)) / ACCUM_STEPS
    total_loss += loss.item()
    loss.backward()
  torch.nn.utils.clip_grad_norm_(session.model.parameters(), max_norm=GRAD_CLIP)
  session.optimizer.step()
  return total_loss


def periodic_log_eval_and_save(session: TrainingSession, step: int, total_loss: float, lr: float) -> None:
  if step % LOG_EVERY == 0:
    cum_h = (session.prior_seconds + time.time() - session.start_time) / 3600
    print(f'step {step}: loss {total_loss:.4f} lr {lr:.2e} cum {cum_h:.2f}h', flush=True)
  if step % EVAL_EVERY == 0:
    val_loss = evaluate(session)
    text = sample_once(session)
    print(f'step {step}: val loss {val_loss:.4f}', flush=True)
    print(f'step {step}: sample {text!r}', flush=True)
    append_sample(step, val_loss, text)
  if step > 0 and step % SAVE_EVERY == 0:
    save_checkpoint(session, step + 1)


def sample_once(session: TrainingSession) -> str:
  session.model.eval()
  text = generate_example(session.model, session.tokenizer, SAMPLE_PROMPT, SAMPLE_MAX_NEW_TOKENS)
  session.model.train()
  return text


def append_sample(step: int, val_loss: float, text: str) -> None:
  SAMPLES_PATH.parent.mkdir(exist_ok=True)
  with open(SAMPLES_PATH, 'a') as f:
    f.write(json.dumps({'step': step, 'val_loss': val_loss, 'ts': time.time(), 'text': text}) + '\n')


def evaluate(session: TrainingSession) -> float:
  device = next(session.model.parameters()).device
  session.model.eval()
  total = 0.0
  with torch.no_grad():
    for _ in range(VAL_BATCHES):
      x, y = get_batch(session.val_ids, BATCH_SIZE, BLOCK_SIZE)
      x, y = x.to(device), y.to(device)
      with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        logits = session.model(x)
        loss = F.cross_entropy(logits.reshape(-1, session.model.config.vocab_size), y.reshape(-1))
      total += loss.item()
  session.model.train()
  return total / VAL_BATCHES


def save_checkpoint(session: TrainingSession, step: int) -> None:
  cum = session.prior_seconds + (time.time() - session.start_time)
  tmp = session.checkpoint_path.parent / (session.checkpoint_path.name + '.tmp')
  torch.save({'model': session.model.state_dict(), 'optimizer': session.optimizer.state_dict(), 'step': step, 'config': session.model.config, 'tokenizer': session.tokenizer, 'cumulative_seconds': cum}, tmp)
  tmp.replace(session.checkpoint_path)


if __name__ == '__main__':
  main()
