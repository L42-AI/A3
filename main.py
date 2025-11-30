from typing import Literal
from collections.abc import Callable

import torch
import numpy as np

import utils.c_types as type
from tqdm import tqdm

from utils import load_imdb, load_imdb_synth, load_xor

# Q1

def get_longest_sequence(x: type.X | None = None) -> int:
    """
    Go through all sequences in the complete dataset
    Find the length of the longest one.
    """
    if x is None:
        (x_train, _), (x_val, _), (_, _), _ = load_imdb(final=True)
        x = x_train + x_val

    return max(len(seq) for seq in x)

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
(x_train, y_train), (x_val, y_val), (i2w, w2i), numcls = load_imdb()
x_train, x_val = pad_sequences(
    x_train,
    x_val,
    w2i.get(".pad"),
    sequence_length
)

def train(
    x_train: type.X,
    y_train: type.Y,
    batch_size: int = 64
) -> None:

    num_batches_train = len(x_train) // batch_size + 1
    num_batches_val = len(x_val) // batch_size + 1

    for batch_idx in range(num_batches_train):
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size
        X = torch.tensor(x_train[start_idx:end_idx], dtype=torch.long)
        T = torch.tensor(y_train[start_idx:end_idx], dtype=torch.long)

    for batch_idx in range(num_batches_val):
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size
        X = torch.tensor(x_val[start_idx:end_idx], dtype=torch.long)
        T = torch.tensor(y_val[start_idx:end_idx], dtype=torch.long)

# Q2

class Baseline(torch.nn.Module):
    def __init__(self, vocab_size: int, num_classes: int):
        super(Baseline, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, 300, padding_idx=0) # Add padding index to ignore .pad token
        self.fc = torch.nn.Linear(300, num_classes)
        self.activation = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        pooled = self.apply_pooling(embedded)
        output = self.fc(pooled)

        return self.activation(output)
    
    def apply_pooling(self, embedded: torch.Tensor) -> torch.Tensor: # TODO: Needs to be padding-aware
        return embedded.mean(dim=1)
    
# Q3

class PoolingBaseline(Baseline):
    def __init__(self, vocab_size: int, num_classes: int, pooling: Literal['mean', 'max', 'first']='mean'):
        super().__init__(vocab_size, num_classes)

        assert pooling in ['mean', 'max', 'first'], "Pooling must be 'mean', 'max' or 'first'"
        self.pooling = pooling

    def apply_pooling(self, embedded: torch.Tensor) -> torch.Tensor: # TODO: Needs to be padding-aware
        if self.pooling == 'mean':
            return embedded.mean(dim=1)
        elif self.pooling == 'max':
            return embedded.max(dim=1)[0]
        elif self.pooling == 'first':
            return embedded[:, 0, :]
    
def run_epoch_on_segment(
    model: Baseline,
    x: type.X,
    y: type.Y,
    batch_size: int,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, float]:
    
    running_loss = 0.0
    running_accuracy = 0.0
    num_batches = len(x) // batch_size + 1
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size
        X = torch.tensor(x[start_idx:end_idx], dtype=torch.long)
        T = torch.tensor(y[start_idx:end_idx], dtype=torch.long)

        Y = model.forward(X)

        loss = torch.nn.functional.cross_entropy(Y, T)
        accuracy = (Y.argmax(dim=1) == T).float().mean()

        running_loss += loss.item()
        running_accuracy += accuracy.item()

        if optimizer is not None:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    return running_loss / num_batches, running_accuracy / num_batches

def train(
    model: Baseline,
    x_train: type.X,
    y_train: type.Y,
    batch_size: int = 64,
    epochs: int = 5,
    lr: float = 0.001,
    plot: bool = False
) -> None:

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses: list[float] = []
    train_accuracies: list[float] = []
    val_losses: list[float] = []
    val_accuracies: list[float] = []

    for epoch in range(epochs):

        train_loss, train_accuracy = run_epoch_on_segment(
            model=model,
            x=x_train,
            y=y_train,
            batch_size=batch_size,
            optimizer=optimizer
        )

        val_loss, val_accuracy = run_epoch_on_segment(
            model=model,
            x=x_val,
            y=y_val,
            batch_size=batch_size,
            optimizer=None
        )

        train_losses.append(train_loss) 
        train_accuracies.append(train_accuracy)
        val_losses.append(val_loss) 
        val_accuracies.append(val_accuracy)

        print(f"    Epoch {epoch+1} | Train Loss: {train_losses[-1]:.4f} | Train Acc: {train_accuracies[-1]:.4f} | Val Loss: {val_losses[-1]:.4f} | Val Acc: {val_accuracies[-1]:.4f}")

    if plot:
        import matplotlib.pyplot as plt
        epochs = np.arange(1, epochs + 1)
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses, label='Train Loss')
        plt.plot(epochs, val_losses, label='Validation Loss')
        plt.title('Loss over Epochs')
        plt.xlabel('Epochs')
        plt.grid(True)
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_accuracies, label='Train Accuracy')
        plt.plot(epochs, val_accuracies, label='Validation Accuracy')
        plt.title('Accuracy over Epochs')
        plt.xlabel('Epochs')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()

# for dataset_func in [load_imdb, load_imdb_synth, load_xor]:
#     print(f"Dataset {dataset_func.__name__}:")
#     dataset_func: Callable[[], tuple[type.Dataset, type.Dataset, tuple[type.I2W, type.W2I], Literal[2]]]
#     (x_train, y_train), (x_val, y_val), (i2w, w2i), numcls = dataset_func()
    
#     x = None if dataset_func.__name__ == "load_imdb" else x_train + x_val # for IMDb, sequence length must be calculated on full dataset
    
#     sequence_length = get_longest_sequence(x)
#     x_train, x_val = pad_sequences(
#         x_train,
#         x_val,
#         w2i.get(".pad"),
#         sequence_length
#     )

#     vocab_size = len(i2w)
#     for pooling_method in ['mean', 'max', 'first']:
#         print(f"Training with {pooling_method} pooling:")
#         model = PoolingBaseline(vocab_size, numcls, pooling=pooling_method)
#         train(model, x_train, y_train, batch_size=64, epochs=5, lr= 0.002, plot=False)

# Q4:

class SimpleSelfAttention(torch.nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_t = x.transpose(1, 2)
        w = (x_t @ x).softmax(dim=-1) # TODO: Check if needs to convert to causal
        return (w @ x_t).transpose(1, 2)
    
class SelfAttention(PoolingBaseline):
    def __init__(self, vocab_size: int, num_classes: int, pooling: Literal['mean', 'max', 'first']='max'):
        super().__init__(vocab_size, num_classes, pooling)

        self.attention = SimpleSelfAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:, :256]  # Ensure input is capped at 256 tokens
        embedded = self.embedding(x)
        attended = self.attention(embedded)
        pooled = self.apply_pooling(attended)
        output = self.fc(pooled)
        return self.activation(output)
    
# Q5:

# for dataset_func in [load_imdb, load_imdb_synth, load_xor]:
#     print(f"Dataset {dataset_func.__name__}:")
#     dataset_func: Callable[[], tuple[type.Dataset, type.Dataset, tuple[type.I2W, type.W2I], Literal[2]]]
#     (x_train, y_train), (x_val, y_val), (i2w, w2i), numcls = dataset_func()
    
#     x = None if dataset_func.__name__ == "load_imdb" else x_train + x_val # for IMDb, sequence length must be calculated on full dataset
    
#     sequence_length = get_longest_sequence(x)
#     x_train, x_val = pad_sequences(
#         x_train,
#         x_val,
#         w2i.get(".pad"),
#         sequence_length
#     )

#     vocab_size = len(i2w)
#     for pooling_method in ['mean', 'max', 'first']:
#         print(f"  Training with {pooling_method} pooling:")
#         model = SelfAttention(vocab_size, numcls, pooling=pooling_method)
#         train(model, x_train, y_train, batch_size=64, epochs=5, lr= 0.001, plot=False)
    
    
for dataset_func in [load_imdb_synth, load_xor]:
    print(f"Dataset {dataset_func.__name__}:")
    dataset_func: Callable[[], tuple[type.Dataset, type.Dataset, tuple[type.I2W, type.W2I], Literal[2]]]
    (x_train, y_train), (x_val, y_val), (i2w, w2i), numcls = dataset_func()
    
    sequence_length = get_longest_sequence(x_train + x_val)
    x_train, x_val = pad_sequences(
        x_train,
        x_val,
        w2i.get(".pad"),
        sequence_length
    )

    vocab_size = len(i2w)
    for pooling_method in ['first']:
        print(f"  Training with {pooling_method} pooling:")
        model = SelfAttention(vocab_size, numcls, pooling=pooling_method)
        train(model, x_train, y_train, batch_size=64, epochs=5, lr= 0.001, plot=False)
    

