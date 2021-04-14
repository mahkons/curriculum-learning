import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FunSeq():
    def __init__(self, dim, num_linear, activations, max_len):
        self.dim = dim
        self.max_len = max_len

        self.f = []

        for act in activations:
            if act == "relu":
                self.f.append(F.relu)
            elif act == "tanh":
                self.f.append(torch.tanh)
            elif act == "sigmoid":
                self.f.append(torch.sigmoid)
            elif act == "id":
                self.f.append(lambda x: x)
            elif act == "sin":
                self.f.append(torch.sin)
            else:
                raise AttributeError("Unknown activation")

        for _ in range(num_linear):
            self.f.append(torch.nn.Linear(self.dim, self.dim))

        self.f = [lambda x: x] + list(np.random.permutation(self.f))
        self.init_val = np.random.uniform(self.dim)
        self.state_sz = len(self.f)

    def generate(self, batch_size, len_seq=None):
        if len_seq is None:
            len_seq = np.random.randint(1, self.max_len + 1, size=batch_size)
        x = np.random.randint(1, self.state_sz, size=(batch_size, self.max_len))
        for i, l in enumerate(len_seq):
            x[i, l:] = 0

        y = torch.ones(batch_size, self.dim) * self.init_val
        with torch.no_grad():
            for i in range(batch_size):
                for op in x[i]:
                    y[i] = self.f[op](y[i])

        return F.one_hot(torch.tensor(x), self.state_sz).float(), y

