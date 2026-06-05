import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from common import BaseClass

class CpuClass(BaseClass):
    def __init__(self, n, Re, u0, dt, dtype):
        super().__init__(n, Re, u0, dt, dtype, device="cpu")
