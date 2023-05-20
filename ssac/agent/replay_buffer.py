from typing import Iterator, Optional

import numpy as np

from ssac.training.trajectory import Transition


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int, batch_size: int):
        self._random = np.random.RandomState(seed)
        self.capacity = capacity
        self.buffer: list[Optional[Transition]] = []
        self.position = 0
        self._batch_size = batch_size

    def store(self, transition: Transition):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = transition
        self.position = int((self.position + 1) % self.capacity)

    def sample(self, num_samples: int) -> Iterator[Transition]:
        for _ in range(num_samples):
            ids = self._random.randint(0, len(self.buffer), self._batch_size)
            batch = np.asarray(self.buffer)[ids]
            out = Transition(*map(np.stack, zip(*batch)))
            yield out

    def __len__(self):
        return len(self.buffer)
