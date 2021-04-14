import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from envs import FunSeq
from logger import init_logger, log

DIM = 100
activations = ["relu", "id", "sin", "tanh", "sigmoid"]
NUM_LINEAR = 25
MAX_LEN = 100
device = torch.device("cuda")
BATCH_SIZE = 64
LR = 1e-3
ITERATIONS = 10000
HIDDEN_SIZE = 64
NUM_LAYERS = 2

class SeqAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(env.state_sz, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, batch_first=True).to(device)
        self.linear = nn.Linear(2 * NUM_LAYERS * HIDDEN_SIZE, DIM)

    def forward(self, x):
        o, (h, c) = self.lstm(x)
        x = torch.cat([h, c], dim=2).permute(1, 0, 2)
        return self.linear(x.reshape(x.shape[0], -1))


# only on MAX_LEN tensors
def train(agent, env):
    optimizer = torch.optim.Adam(agent.parameters(), lr=LR)
    
    sum_loss = 0.
    cnt = 0.
    log().add_plot("loss", columns=("iteration", "loss"))
    for i in range(ITERATIONS):
        x, y = env.generate(BATCH_SIZE, len_seq=torch.ones(BATCH_SIZE, dtype=torch.long) * MAX_LEN)
        x, y = x.to(device), y.to(device)
        output = agent(x)

        loss = F.mse_loss(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        log().add_plot_point("loss", (i, loss.item()))

        sum_loss += loss.item()
        cnt += 1

        if i % 100 == 0:
            print("Iteration {}. Loss {}".format(i, sum_loss / cnt))
            sum_loss = 0.
            cnt = 0.
    log().save_logs()

# increasing len size
def train_curriculum(agent, env):
    optimizer = torch.optim.Adam(agent.parameters(), lr=LR)
    
    sum_loss = 0.
    cnt = 0.
    sum_val_loss = 0.
    log().add_plot("loss", columns=("iteration", "loss"))
    for i in range(ITERATIONS):
        mx_len = min(MAX_LEN, 1 + (2 * MAX_LEN * i) // ITERATIONS)
        mn_len = 1 + (MAX_LEN * i) // ITERATIONS
        x, y = env.generate(BATCH_SIZE, len_seq=np.random.randint(mn_len, mx_len + 1, size=BATCH_SIZE))
        x, y = x.to(device), y.to(device)
        output = agent(x)

        loss = F.mse_loss(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sum_loss += loss.item()
        cnt += 1

        with torch.no_grad():
            x, y = env.generate(BATCH_SIZE, len_seq=torch.ones(BATCH_SIZE, dtype=torch.long) * MAX_LEN)
            x, y = x.to(device), y.to(device)
            output = agent(x)
            val_loss = F.mse_loss(output, y)
            sum_val_loss += val_loss.item()
        log().add_plot_point("loss", (i, val_loss.item()))

        if i % 100 == 0:
            print("Iteration {}. Loss {}. Val Loss {}".format(i, sum_loss / cnt, sum_val_loss / cnt))
            sum_loss = 0.
            sum_val_loss = 0.
            cnt = 0.
    log().save_logs()

if __name__ == "__main__":
    np.random.seed(23)
    torch.manual_seed(23)
    init_logger("logdir", "no-curriculum-long")

    env = FunSeq(DIM, NUM_LINEAR, activations, MAX_LEN)
    agent = SeqAgent().to(device)
    train(agent, env)
    #  train_curriculum(agent, env)
