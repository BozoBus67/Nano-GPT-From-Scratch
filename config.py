from dataclasses import dataclass

from model_variants import Gelu_MLP, LayerNorm, RMSNorm, SwiGLU_MLP
from validators import check_divisible


@dataclass
class TransformerConfig:
  vocab_size: int
  d_model: int
  n_blocks: int
  max_seq_len: int
  n_heads: int
  n_kv_heads: int
  mlp_class: type
  norm_class: type

  def __post_init__(self) -> None:
    check_divisible(self.d_model, self.n_heads, 'd_model', 'n_heads')
    check_divisible(self.n_heads, self.n_kv_heads, 'n_heads', 'n_kv_heads')


def small_mha(vocab_size: int) -> TransformerConfig:
  return TransformerConfig(vocab_size=vocab_size, d_model=64, n_blocks=2, max_seq_len=32, n_heads=4, n_kv_heads=4, mlp_class=SwiGLU_MLP, norm_class=RMSNorm)


def small_gqa(vocab_size: int) -> TransformerConfig:
  return TransformerConfig(vocab_size=vocab_size, d_model=64, n_blocks=2, max_seq_len=32, n_heads=4, n_kv_heads=2, mlp_class=SwiGLU_MLP, norm_class=RMSNorm)


def small_mqa(vocab_size: int) -> TransformerConfig:
  return TransformerConfig(vocab_size=vocab_size, d_model=64, n_blocks=2, max_seq_len=32, n_heads=4, n_kv_heads=1, mlp_class=SwiGLU_MLP, norm_class=RMSNorm)


def small_gelu(vocab_size: int) -> TransformerConfig:
  return TransformerConfig(vocab_size=vocab_size, d_model=64, n_blocks=2, max_seq_len=32, n_heads=4, n_kv_heads=4, mlp_class=Gelu_MLP, norm_class=RMSNorm)


def small_layernorm(vocab_size: int) -> TransformerConfig:
  return TransformerConfig(vocab_size=vocab_size, d_model=64, n_blocks=2, max_seq_len=32, n_heads=4, n_kv_heads=4, mlp_class=SwiGLU_MLP, norm_class=LayerNorm)


def medium_mha(vocab_size: int) -> TransformerConfig:
  return TransformerConfig(vocab_size=vocab_size, d_model=256, n_blocks=6, max_seq_len=128, n_heads=8, n_kv_heads=8, mlp_class=SwiGLU_MLP, norm_class=RMSNorm)


def medium_gqa(vocab_size: int) -> TransformerConfig:
  return TransformerConfig(vocab_size=vocab_size, d_model=256, n_blocks=6, max_seq_len=128, n_heads=8, n_kv_heads=2, mlp_class=SwiGLU_MLP, norm_class=RMSNorm)


def large_gqa(vocab_size: int) -> TransformerConfig:
  return TransformerConfig(vocab_size=vocab_size, d_model=384, n_blocks=8, max_seq_len=512, n_heads=8, n_kv_heads=2, mlp_class=SwiGLU_MLP, norm_class=RMSNorm)
