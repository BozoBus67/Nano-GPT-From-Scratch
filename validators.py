def check_divisible(a: int, b: int, name_a: str, name_b: str) -> None:
  if a % b != 0:
    raise ValueError(f"{name_a} ({a}) must be divisible by {name_b} ({b})")
