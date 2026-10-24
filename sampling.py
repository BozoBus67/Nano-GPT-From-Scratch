import torch


def sample(logits, temperature: float = 1.0) -> int: # 1dtensor, float -> int
  if temperature == 0:
    return int(torch.argmax(logits).item())
  scaled = logits / temperature
  probs = torch.softmax(scaled, dim=-1)
  return int(torch.multinomial(probs, num_samples=1).item())
