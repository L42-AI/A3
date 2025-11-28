from typing import Literal

import torch


import utils.c_types as type
from tqdm import tqdm

from utils import load_imdb

def get_longest_sequence() -> int:
    """
    Go through all sequences in the complete dataset
    Find the length of the longest one.
    """
    (x_train, _), (x_val, _), (_, _), _ = load_imdb(final=True)
    return max(len(seq) for seq in x_train + x_val)


def pad_sequences(x_train: type.X, x_val: type.X, pad_token_id: int, longest_seq: int) -> tuple[type.X, type.X]:
    """
    Pad all sequences in x_train and x_val to the length of longest_seq
    using the pad_token_id.
    Returns the padded x_train and x_val.
    """
    x_train_padded = [seq + [pad_token_id] * (longest_seq - len(seq)) for seq in x_train]
    x_val_padded = [seq + [pad_token_id] * (longest_seq - len(seq)) for seq in x_val]
    return x_train_padded, x_val_padded

sequence_length = get_longest_sequence()
(x_train, y_train), (x_val, y_val), (i2w, w2i), numcls = load_imdb(final=False)
x_train, x_val = pad_sequences(
    x_train,
    x_val,
    w2i.get(".pad"),
    sequence_length
)

def train(model: torch.nn.Module, x_train: type.X, y_train: type.Y, batch_size: int = 64):

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_batches_train = len(x_train) // batch_size + 1
    num_batches_val = len(x_val) // batch_size + 1

    for batch_idx in range(num_batches_train):
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size
        X = torch.tensor(x_train[start_idx:end_idx], dtype=torch.long)
        T = torch.tensor(y_train[start_idx:end_idx], dtype=torch.long)

        Y = model.forward(X)

        loss = torch.nn.functional.cross_entropy(Y, T)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx+1}/{num_batches_train}, Loss: {loss.item():.4f}")
    print(f"Batch {batch_idx+1}/{num_batches_train}, Loss: {loss.item():.4f}")



    # for batch_idx in range(num_batches_val):
    #     start_idx = batch_idx * batch_size
    #     end_idx = start_idx + batch_size
    #     X = torch.tensor(x_val[start_idx:end_idx], dtype=torch.long)
    #     T = torch.tensor(y_val[start_idx:end_idx], dtype=torch.long)

    #     Y = model.forward(X)


class Baseline(torch.nn.Module):
    def __init__(self, vocab_size: int, num_classes: int, pooling: Literal['mean', 'max', 'first']='mean'):
        super(Baseline, self).__init__()
        assert pooling in ['mean', 'max', 'first'], "Pooling must be 'mean', 'max' or 'first'"
        self.pooling = pooling
        self.embedding = torch.nn.Embedding(vocab_size, 300)
        self.fc = torch.nn.Linear(300, num_classes)
        self.activation = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        if self.pooling == 'mean':
            pooled = embedded.mean(dim=1)
        elif self.pooling == 'max':
            pooled, _ = embedded.max(dim=1)
        else:  # 'first'
            pooled = embedded[:, 0, :]
        output = self.fc(pooled)

        return self.activation(output)
    
vocab_size = len(i2w)

for pooling_method in ['mean', 'max', 'first']:
    print(f"Training with {pooling_method} pooling:")
    model = Baseline(vocab_size, numcls, pooling=pooling_method)
    train(model, x_train, y_train, batch_size=64)

    

