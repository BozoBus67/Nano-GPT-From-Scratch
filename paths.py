from pathlib import Path


CORPUS_PATH = Path(__file__).parent / 'data' / 'tinystories.txt'
CORPUS_PATH_FULL = Path(__file__).parent / 'data' / 'tinystories_full.txt'
PREPARED_DATA_PATH = Path(__file__).parent / 'data' / 'tinystories_full.prepared.pt'
CHECKPOINT_PATH = Path(__file__).parent / 'checkpoints' / '2026-10-20_pretrain_m13M_s512.pt'
