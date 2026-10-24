import torch
import torch.nn as nn
import torch.nn.functional as F

from validators import check_divisible


LN_EPS = 1e-5      # avoid divide-by-zero if a row's variance is 0
INIT_SCALE = 0.02  # GPT-2 style weight init std
ROPE_BASE = 10000  # frequency base for RoPE rotation angles


class Transformer(nn.Module):
  def __init__(self, config) -> None:
    super().__init__()
    self.config = config
    self.token_embed = nn.Parameter(torch.randn(config.vocab_size, config.d_model) * INIT_SCALE)
    self.blocks = nn.ModuleList([Block(config.d_model, config.n_heads, config.max_seq_len, n_kv_heads=config.n_kv_heads, mlp_class=config.mlp_class, norm_class=config.norm_class) for _ in range(config.n_blocks)])
    self.norm_f = config.norm_class(config.d_model)

  def forward(self, ids): # 2dtensor -> 3dtensor
    x = self.token_embed[ids]
    for block in self.blocks:
      x = block(x)
    x = self.norm_f(x)
    return x @ self.token_embed.T

  def enable_cache(self) -> None:
    device = next(self.parameters()).device
    for block in self.blocks:
      attn = block.attn
      attn.cache = KVCache(self.config.max_seq_len, attn.n_kv_heads, attn.d_head).to(device)

  def clear_cache(self) -> None:
    for block in self.blocks:
      block.attn.cache = None


class Block(nn.Module):
  def __init__(self, d_model: int, n_heads: int, max_seq_len: int, n_kv_heads: int, mlp_class: type, norm_class: type) -> None:
    super().__init__()
    self.norm1 = norm_class(d_model)
    self.attn = Attention(d_model, n_heads, max_seq_len, n_kv_heads=n_kv_heads)
    self.norm2 = norm_class(d_model)
    self.mlp = mlp_class(d_model)

  def forward(self, x): # 3dtensor -> 3dtensor
    x = x + self.attn(self.norm1(x))
    x = x + self.mlp(self.norm2(x))
    return x


class Attention(nn.Module):  # GQA/MHA/MQA via n_kv_heads
  def __init__(self, d_model: int, n_heads: int, max_seq_len: int, n_kv_heads: int) -> None:
    super().__init__()
    check_divisible(d_model, n_heads, 'd_model', 'n_heads')
    check_divisible(n_heads, n_kv_heads, 'n_heads', 'n_kv_heads')
    self.n_heads = n_heads
    self.n_kv_heads = n_kv_heads
    self.d_head = d_model // n_heads
    self.w_q = nn.Parameter(torch.randn(d_model, n_heads * self.d_head) * INIT_SCALE)
    self.w_k = nn.Parameter(torch.randn(d_model, n_kv_heads * self.d_head) * INIT_SCALE)
    self.w_v = nn.Parameter(torch.randn(d_model, n_kv_heads * self.d_head) * INIT_SCALE)
    self.w_o = nn.Parameter(torch.randn(d_model, d_model) * INIT_SCALE)
    self.rope = RoPE(self.d_head, max_seq_len)
    self.cache = None  # set by Transformer.enable_cache() for inference

  def forward(self, x): # 3dtensor -> 3dtensor
    B, T, D = x.shape
    Q = self._project(x, self.w_q, self.n_heads)     # (B, H, T, d_head)
    K = self._project(x, self.w_k, self.n_kv_heads)  # (B, n_kv, T, d_head)
    V = self._project(x, self.w_v, self.n_kv_heads)  # (B, n_kv, T, d_head)
    pos_offset = self.cache.pos if self.cache is not None else 0
    Q = self.rope(Q, pos_offset=pos_offset)
    K = self.rope(K, pos_offset=pos_offset)
    if self.cache is not None:
      K, V = self.cache.append(K, V)
    is_causal = Q.shape[-2] == K.shape[-2]
    out = F.scaled_dot_product_attention(Q, K, V, is_causal=is_causal, enable_gqa=True)  # (B, H, T, d_head)
    out = out.transpose(1, 2).reshape(B, T, D)       # (B, T, D)
    return out @ self.w_o

  def _project(self, x, W, n_heads): # 3dtensor, 2dtensor, int -> 4dtensor
    B, T, _ = x.shape
    return (x @ W).reshape(B, T, n_heads, self.d_head).transpose(1, 2)


class RoPE(nn.Module):
  def __init__(self, d_head: int, max_seq_len: int) -> None:
    super().__init__()
    check_divisible(d_head, 2, 'd_head', '2')
    inv_freq = 1.0 / (ROPE_BASE ** (torch.arange(0, d_head, 2).float() / d_head))  # (d_head/2,)
    positions = torch.arange(max_seq_len).float()  # (max_seq_len,)
    angles = positions[:, None] * inv_freq[None, :]  # (max_seq_len, d_head/2)
    self.register_buffer('cos', torch.cos(angles))
    self.register_buffer('sin', torch.sin(angles))

  def forward(self, x, pos_offset: int = 0): # 4dtensor, int -> 4dtensor
    T = x.shape[-2]
    cos = self.cos[pos_offset:pos_offset+T]  # (T, d_head/2)
    sin = self.sin[pos_offset:pos_offset+T]  # (T, d_head/2)
    x_even = x[..., 0::2]  # (B, H, T, d_head/2)
    x_odd = x[..., 1::2]   # (B, H, T, d_head/2)
    rot_even = x_even * cos - x_odd * sin
    rot_odd = x_even * sin + x_odd * cos
    return torch.stack([rot_even, rot_odd], dim=-1).flatten(-2)  # (B, H, T, d_head)


class KVCache(nn.Module):
  def __init__(self, max_seq_len: int, n_kv_heads: int, d_head: int) -> None:
    super().__init__()
    self.register_buffer('k', torch.zeros(1, n_kv_heads, max_seq_len, d_head), persistent=False)
    self.register_buffer('v', torch.zeros(1, n_kv_heads, max_seq_len, d_head), persistent=False)
    self.pos = 0

  def append(self, k_new, v_new): # 4dtensor, 4dtensor -> 4dtensor, 4dtensor
    T = k_new.shape[-2]
    self.k[:, :, self.pos:self.pos+T] = k_new
    self.v[:, :, self.pos:self.pos+T] = v_new
    self.pos += T
    return self.k[:, :, :self.pos], self.v[:, :, :self.pos]
