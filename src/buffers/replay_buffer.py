# src/buffers/replay_buffer.py


from dataclasses import dataclass
from typing import List, Tuple, Optional
import random
import torch


@dataclass
class BufItem:
    x: torch.Tensor
    y: int
    score: float = 1.0


class RingBuffer:
    """
    FIFO buffer (cyclic).
    Maintains recency bias via sliding window.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.items: List[BufItem] = []
        self.pos = 0

    def __len__(self) -> int:
        return len(self.items)

    def add(self, x: torch.Tensor, y: int) -> None:
        item = BufItem(x.detach().cpu(), int(y))

        if len(self.items) < self.capacity:
            self.items.append(item)
        else:
            self.items[self.pos] = item
            self.pos = (self.pos + 1) % self.capacity

    def sample_uniform(self, k: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        k = min(k, len(self.items))
        if k == 0:
            return None, None

        batch = random.sample(self.items, k)
        x = torch.stack([b.x for b in batch])
        y = torch.tensor([b.y for b in batch], dtype=torch.long)

        return x, y


class ReservoirBuffer:
    """
    Reservoir sampling buffer.

    Guarantees equal probability retention for all seen samples.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.items: List[BufItem] = []
        self.seen = 0

    def __len__(self) -> int:
        return len(self.items)

    def add(self, x: torch.Tensor, y: int) -> None:
        self.seen += 1
        item = BufItem(x.detach().cpu(), int(y))

        if len(self.items) < self.capacity:
            self.items.append(item)
        else:
            j = random.randrange(self.seen)
            if j < self.capacity:
                self.items[j] = item

    def sample_uniform(self, k: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        k = min(k, len(self.items))
        if k == 0:
            return None, None

        batch = random.sample(self.items, k)
        x = torch.stack([b.x for b in batch])
        y = torch.tensor([b.y for b in batch], dtype=torch.long)

        return x, y

    def sample_weighted(self, k: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Weighted sampling based on score.

        NOTE: Sampling is WITH replacement.
        This increases prioritization strength but may reduce diversity.
        """
        k = min(k, len(self.items))
        if k == 0:
            return None, None

        weights = [max(1e-6, b.score) for b in self.items]
        indices = random.choices(range(len(self.items)), weights=weights, k=k)

        batch = [self.items[i] for i in indices]
        x = torch.stack([b.x for b in batch])
        y = torch.tensor([b.y for b in batch], dtype=torch.long)

        return x, y
