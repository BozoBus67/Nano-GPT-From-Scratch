from datasets import load_dataset

from paths import CORPUS_PATH_FULL


DATASET_NAME = 'roneneldan/TinyStories'


def main() -> None:
  if CORPUS_PATH_FULL.exists():
    print(f'{CORPUS_PATH_FULL} already exists; delete it to re-download.', flush=True)
    return
  CORPUS_PATH_FULL.parent.mkdir(parents=True, exist_ok=True)
  print(f'Downloading {DATASET_NAME} to {CORPUS_PATH_FULL}', flush=True)
  ds = load_dataset(DATASET_NAME, split='train')
  print(f'{len(ds)} examples', flush=True)
  total_bytes = 0
  with open(CORPUS_PATH_FULL, 'w') as f:
    for i, ex in enumerate(ds):
      text = ex['text']
      f.write(text)
      if not text.endswith('<|endoftext|>'):
        f.write('<|endoftext|>')
      f.write('\n')
      total_bytes += len(text)
      if i > 0 and i % 100000 == 0:
        print(f'  {i}/{len(ds)} examples, {total_bytes/1024/1024:.1f} MB', flush=True)
  print(f'Done. {total_bytes/1024/1024:.1f} MB', flush=True)


if __name__ == '__main__':
  main()
