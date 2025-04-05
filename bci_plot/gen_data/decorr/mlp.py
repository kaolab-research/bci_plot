
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, odim=2, hdim=128, device='cpu'):
        super().__init__()
        #self.mlp = nn.Sequential(nn.LazyLinear(hdim), nn.GELU(), nn.Linear(hdim, odim))
        self.mlp = nn.Sequential(nn.LazyLinear(hdim), nn.ReLU(), nn.Linear(hdim, odim))
        self.mseloss = nn.MSELoss()
        self.device = device
        self.to(device)
        return
    def forward(self, x, flatten=False):
        if flatten:
            x = x.flatten(start_dim=1)
        return self.mlp(x)
    def predict(self, X, batch_size=64):
        self.eval()
        if batch_size is None:
            batch_size = len(X)
        if not torch.is_tensor(X):
            X = torch.tensor(X, dtype=torch.float).to(self.device)
        else:
            X = X.to(self.device)
        with torch.no_grad():
            K = int(np.ceil(X.shape[0]/batch_size))
            out = torch.cat([self(X[batch_size*i:batch_size*(i+1)], flatten=True) for i in range(K)], axis=0)
        return out