import random
import numpy as np
from torch.utils.data import Sampler
from typing import List

class StratifiedBatchSampler(Sampler[List[int]]):

    def __init__(self, dataset_ids: List[int], batch_size: int): 
        self.batch_size = batch_size
        self.groups = {}
        for idx, ds in enumerate(dataset_ids):
            self.groups.setdefault(ds, []).append(idx)

        sizes = np.array([len(v) for v in self.groups.values()], dtype=float)
        self.group_ids = list(self.groups.keys())
        self.probs = sizes / sizes.sum()

    def __iter__(self):
        n_batches = len(self)

        buffers = {ds: idxs.copy() for ds, idxs in self.groups.items()}
        for buf in buffers.values():
            random.shuffle(buf)

        for _ in range(n_batches):
            ds = random.choices(self.group_ids, weights=self.probs, k=1)[0]
            buf = buffers[ds]
            if len(buf) < self.batch_size:
                buf.extend(self.groups[ds])
                random.shuffle(buf)

            batch = [buf.pop() for _ in range(self.batch_size)]
            yield batch

    def __len__(self) -> int:
        total = sum(len(v) for v in self.groups.values())
        return total // self.batch_size
