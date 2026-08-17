import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from common import BaseClass

class GpuClass(BaseClass):
    def __init__(self, n, Re, u0, dt, dtype):
        super().__init__(n, Re, u0, dt, dtype, device="cuda")

    def to_cpu(self):
        return (self.u.cpu().numpy(), self.v.cpu().numpy(), self.p.cpu().numpy())

    def synchronize(self):
        torch.cuda.synchronize()
