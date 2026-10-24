import torch
import torch.nn as nn

from model import INIT_SCALE, LN_EPS


class Gelu_MLP(nn.Module):
  def __init__(self, d_model: int) -> None:
    super().__init__()
    self.w_up = nn.Parameter(torch.randn(d_model, 4 * d_model) * INIT_SCALE)
    self.w_down = nn.Parameter(torch.randn(4 * d_model, d_model) * INIT_SCALE)
    self.act = Gelu()

  def forward(self, x): # 3dtensor -> 3dtensor
    h = x @ self.w_up
    h = self.act(h)
    return h @ self.w_down


class Gelu(nn.Module):
  def forward(self, x): # 3dtensor -> 3dtensor
    return 0.5 * x * (1 + torch.tanh((2 / torch.pi) ** 0.5 * (x + 0.044715 * x ** 3)))


class SwiGLU_MLP(nn.Module):
  def __init__(self, d_model: int) -> None:
    super().__init__()
    hidden = int(8 * d_model / 3)  # matches plain MLP param count when hidden = (2/3) * 4D
    self.w_gate = nn.Parameter(torch.randn(d_model, hidden) * INIT_SCALE)
    self.w_up = nn.Parameter(torch.randn(d_model, hidden) * INIT_SCALE)
    self.w_down = nn.Parameter(torch.randn(hidden, d_model) * INIT_SCALE)
    self.act = Swish()

  def forward(self, x): # 3dtensor -> 3dtensor
    return (self.act(x @ self.w_gate) * (x @ self.w_up)) @ self.w_down


class Swish(nn.Module):
  def forward(self, x): # 3dtensor -> 3dtensor
    return x * 0.5 * (1 + torch.tanh(x / 2))


class RMSNorm(nn.Module):
  def __init__(self, d_model: int) -> None:
    super().__init__()
    self.gamma = nn.Parameter(torch.ones(d_model))

  def forward(self, x): # 3dtensor -> 3dtensor
    rms = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + LN_EPS)
    return x / rms * self.gamma


class LayerNorm(nn.Module):
  def __init__(self, d_model: int) -> None:
    super().__init__()
    self.gamma = nn.Parameter(torch.ones(d_model))
    self.beta = nn.Parameter(torch.zeros(d_model))

  def forward(self, x): # 3dtensor -> 3dtensor
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    x_hat = (x - mean) / torch.sqrt(var + LN_EPS)
    return x_hat * self.gamma + self.beta
