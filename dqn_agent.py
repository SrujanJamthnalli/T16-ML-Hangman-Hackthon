import torch, torch.nn as nn, torch.optim as optim, numpy as np, random
from collections import deque

class DQNNet(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, action_size)
        )
    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    def push(self, s, a, r, ns, d):
        self.buffer.append((s,a,r,ns,d))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s,a,r,ns,d = zip(*batch)
        return np.stack(s), np.array(a), np.array(r), np.stack(ns), np.array(d)
    def __len__(self): return len(self.buffer)

def select_action(net, state_vec, guessed, eps, alphabet):
    with torch.no_grad():
        q = net(torch.from_numpy(state_vec).unsqueeze(0).float()).squeeze(0).numpy()
    for i,a in enumerate(alphabet):
        if a in guessed:
            q[i] = -1e9
    if random.random() < eps:
        legal = [i for i,a in enumerate(alphabet) if a not in guessed]
        return random.choice(legal)
    return int(np.argmax(q))
