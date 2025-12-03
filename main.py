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

# sequence_length = get_longest_sequence()
# (x_train, y_train), (x_val, y_val), (i2w, w2i), numcls = load_imdb()
# x_train, x_val = pad_sequences(
#     x_train,
#     x_val,
#     w2i.get(".pad"),
#     sequence_length
# )

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
    def __init__(self, vocab_size: int, num_classes: int, pooling: Literal['mean', 'max', 'first']='mean'):
        super(Baseline, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, 300, padding_idx=0) # Add padding index to ignore .pad token
        self.fc = torch.nn.Linear(300, num_classes)

        assert pooling in ['mean', 'max', 'first'], "Pooling must be 'mean', 'max' or 'first'"
        if pooling == 'mean':
            self.apply_pooling = self._apply_mean_pooling
        elif pooling == 'max':
            self.apply_pooling = self._apply_max_pooling
        elif pooling == 'first':
            self.apply_pooling = lambda embedded: embedded[:, 0, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        pooled = self.apply_pooling(embedded)
        return self.fc(pooled)
    
    def find_mask(self, x: torch.Tensor) -> torch.Tensor:
        """ all non-zero sums are real embeddings, zeros are padding """
        return (x.abs().sum(dim=-1) != 0).unsqueeze(-1)  # Assuming padding index is 0

    def _apply_mean_pooling(self, embedded: torch.Tensor) -> torch.Tensor:
        non_pad_mask = self.find_mask(embedded) # Create a mask for non-padding tokens
        summed = (embedded * non_pad_mask).sum(dim=1)  # Take sum of real sequence tokens
        counts = non_pad_mask.sum(dim=1)  # Take count of real sequence tokens
        return summed / counts.clamp(min=1)  # Avoid division by zero
    
    def _apply_max_pooling(self, embedded: torch.Tensor) -> torch.Tensor:
        non_pad_mask = self.find_mask(embedded) # Create a mask for non-padding tokens
        masked_embedded = embedded.masked_fill(~non_pad_mask, float('-inf'))  # Set padded positions to -inf for max pooling
        return masked_embedded.max(dim=1)[0]  # Max pooling over the sequence length dimension
    

# Q3

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
#         print(f"  Training with {pooling_method} pooling:")
#         model = Baseline(vocab_size, numcls, pooling=pooling_method)
#         train(model, x_train, y_train, batch_size=64, epochs=5, lr= 0.002, plot=False)
# Q4:

class SimpleSelfAttention(torch.nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_t = x.transpose(-2, -1)
        w = (x_t @ x).softmax(dim=-1)
        return (w @ x_t).transpose(-2, -1)
    
class SelfAttentionNN(Baseline):
    def __init__(self, vocab_size: int, num_classes: int, pooling: Literal['mean', 'max', 'first']='first'):
        super().__init__(vocab_size, num_classes, pooling)

        self.attention = SimpleSelfAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:, :256]  # Ensure input is capped at 256 tokens
        embedded = self.embedding(x)
        attended = self.attention(embedded)
        pooled = self.apply_pooling(attended)
        return self.fc(pooled)
    
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
#     print(f"  Training with first pooling:")
#     model = SelfAttentionNN(vocab_size, numcls, pooling='first')
#     train(model, x_train, y_train, batch_size=64, epochs=5, lr= 0.002, plot=False)
    
    
# TODO: seems to find perfect accuracy on both with first pooling, so skipping for now 
# TODO: Implement optuna parameter tuning.
# for dataset_func in [load_imdb_synth, load_xor]: 
#     print(f"Dataset {dataset_func.__name__}:")
#     dataset_func: Callable[[], tuple[type.Dataset, type.Dataset, tuple[type.I2W, type.W2I], Literal[2]]]
#     (x_train, y_train), (x_val, y_val), (i2w, w2i), numcls = dataset_func()
    
#     sequence_length = get_longest_sequence(x_train + x_val)
#     x_train, x_val = pad_sequences(
#         x_train,
#         x_val,
#         w2i.get(".pad"),
#         sequence_length
#     )

#     vocab_size = len(i2w)
#     print(f"  Training with first pooling:")
#     model = SelfAttentionNN(vocab_size, numcls, pooling='first')
#     train(model, x_train, y_train, batch_size=64, epochs=5, lr= 0.002, plot=False)
    
# Q6

class SelfAttentionHead(torch.nn.Module):

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.key_head = torch.nn.Linear(embedding_dim, embedding_dim)
        self.query_head = torch.nn.Linear(embedding_dim, embedding_dim)
        self.value_head = torch.nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        keys = self.key_head(x).transpose(-2, -1) # (B, K, T, E) -> (B, K, E, T)
        queries = self.query_head(x).transpose(-2, -1) # (B, K, T, E) -> (B, K, E, T)
        values = self.value_head(x).transpose(-2, -1) # (B, K, T, E) -> (B, K, E, T)

        w_prime = (queries @ keys.transpose(-2, -1)) # (B, K, E, T) @ (B, K, T, E) -> (B, K, E, E)
        scaled_w_prime = w_prime / np.sqrt(self.embedding_dim)
        w = scaled_w_prime.softmax(dim=-1)
        return (w @ values).transpose(-2, -1) # (B, K, E, E) @ (B, K, E, T) -> (B, K, E, T) -> (B, K, T, E)
    
class MultiHeadSelfAttention(torch.nn.Module):
    
    def __init__(self, embedding_dim: int, num_heads: int):
        super().__init__()
        assert embedding_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        self.heads = torch.nn.ModuleList([
            SelfAttentionHead(embedding_dim // num_heads) for _ in range(num_heads)
        ])
        self.final_layer = torch.nn.Linear(embedding_dim, embedding_dim)

    def extract_head_dimension(self, x: torch.Tensor, per_head_dim: int) -> int:
        """
        Reshapes the embedding dimension to have heads as a separate dimension.
        and reorders the dimensions to have the important ones for computation at the end.
        (B, T, E) -> (B, H, T, E/H)
        """
        x = x.view(x.size(0), x.size(1), len(self.heads), per_head_dim)
        return x.transpose(1, 2).contiguous()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        per_head_dim = x.size(-1) // len(self.heads)

        x = self.extract_head_dimension(x, per_head_dim)  # (B, T, E) -> (B, H, T, E/H)
        heads_output = []
        for head in self.heads:
            heads_output.append(head(x))
        unified_heads_output = torch.cat(heads_output, dim=-1)
        return self.final_layer(unified_heads_output)

class FullSelfAttentionNN(SelfAttentionNN):
    def __init__(self, vocab_size: int, num_classes: int, pooling: Literal['mean', 'max', 'first']='first'):
        super().__init__(vocab_size, num_classes, pooling)

        self.attention = MultiHeadSelfAttention(embedding_dim=300, num_heads=6)

for dataset_func in [ load_imdb_synth, load_xor]: 
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
    print(f"  Training with first pooling:")
    model = FullSelfAttentionNN(vocab_size, numcls, pooling='first')
    train(model, x_train, y_train, batch_size=64, epochs=5, lr= 0.002, plot=False)

# Q7:

# Q8:

#TODO: Add positional encoding to Self-Attention models

# Q9:

#TODO: Build into transformer architecture

