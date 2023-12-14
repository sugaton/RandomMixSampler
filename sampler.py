from typing import Iterator, Optional
from torch.utils.data import WeightedRandomSampler, Sampler, SubsetRandomSampler
import torch

class RandomMixSampler(Sampler[int]):
    @staticmethod
    def inrange(l, r):
        def _function(val):
            return l <= val and val < r
        return _function
        
    def __init__(self, 
                 num_samples: Optional[int]= None,
                 generator=None,
                 samplers = list[Sampler[int]]) -> None:
        self.range_func = []
        _total = 0
        _last = 0
        self.samplers = samplers
        for sampler in samplers:
            _total += len(sampler)
            self.range_func.append(self.inrange(_last, _total))
            _last = _total
        if num_samples is None:
            self.num_samples = _total
        else:
            self.num_samples = num_samples

    def _getter(self, idx):
        for item in self.samplers[idx]:  
            yield item

    def __iter__(self) -> Iterator[int]:
        samplers = [self._getter(i) for i in range(len(self.samplers))]
        for _idx in torch.randperm(self.num_samples).tolist():
            cand = next(i for i, f in enumerate(self.range_func) if f(_idx))
            yield next(samplers[cand])

    def __len__(self):
        return self.num_samples


if __name__ == "__main__":
    l=[1. if i % 2==0 else 0.1 for i in range(200)] 
    t=torch.tensor(l)
    m = torch.where(t >= 1.)
    weights = torch.Tensor(l)
    weights[m] = 0

    sampler = WeightedRandomSampler(weights, num_samples=50, replacement=False)
    sampler2 = SubsetRandomSampler(m[0].tolist())

    newsampler = RandomMixSampler(samplers=(sampler, sampler2))
    ids= []
    for idx in newsampler:
        ids.append(idx)

    assert len([i for i in ids if t[i] == 1.]) == 100
