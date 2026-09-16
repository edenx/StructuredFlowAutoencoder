
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
import torch.nn.utils as nn_utils
import math
from typing import List


def sum_except_batch(x):
    return x.view(x.size(0), -1).sum(-1)

def log_normal(x: Tensor) -> Tensor:
    return -(x.square() + math.log(2 * math.pi)).sum(dim=-1) / 2

class ShiftedTanh(torch.nn.Module):
    def __init__(self, a=5, b=0.5):
        super().__init__()
        self.a = a
        self.b = b

    def forward(self, x):
        return torch.tanh(x) * self.a + self.b

def log_normal(x: Tensor) -> Tensor:
    return -(x.square() + math.log(2 * math.pi)).sum(dim=-1) / 2

class SoftplusParameterization(nn.Module):
    def forward(self, X):
        # return torch.exp(X)
        return nn.functional.softplus(X)

class ScaledSigmoid(nn.Module):
    def __init__(self):
        super(ScaledSigmoid, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return 2 * self.sigmoid(x) - 1

def scaledsigmoid(x):
    return torch.sigmoid(x) * 2 - 1


class SineActivation(torch.nn.Module):
    def __init__(self, omega=10):
        super().__init__()
        self.omega = omega

    def forward(self, x):
        return torch.sin(self.omega * x)
    
    
def to_one_hot(z, num_classes):
    """
    Transforms a batch of categorical values into one-hot encoded form.
    
    Args:
    - z (torch.Tensor): A tensor of shape (batch_size,) with categorical values (integers).
    - num_classes (int): The number of unique categories/classes.
    
    Returns:
    - torch.Tensor: A tensor of shape (batch_size, num_classes) representing the one-hot encoded values.
    """
    # Ensure the input batch is a tensor
    if not isinstance(z, torch.Tensor):
        z = torch.tensor(z)
    
    # Create a one-hot encoding of the batch
    one_hot = torch.zeros(z.size(0), num_classes)
    one_hot.scatter_(1, z.unsqueeze(1), 1)
    return one_hot


class MLP(nn.Sequential):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: List[int] = [64, 64],
        fct=nn.Tanh(),
        batch_norm=False,
        dropout=False,
        weight_norm=False,
        layer_norm=False,
        p=0.2
        # fct=ScaledSigmoid()
    ):
        layers = []

        for a, b in zip(
            (in_features, *hidden_features),
            (*hidden_features, out_features),
        ):  
            linear_layer = nn.Linear(a, b)
            if weight_norm:
                linear_layer = nn_utils.weight_norm(linear_layer)
            if batch_norm:
                layers.extend([linear_layer, nn.BatchNorm1d(b), fct])
            elif layer_norm:
                layers.extend([linear_layer, nn.LayerNorm(b), fct])
            elif dropout:
                layers.extend([linear_layer, nn.Dropout(p=p), fct])
            else:
                layers.extend([linear_layer, fct])

        if not weight_norm or batch_norm or layer_norm or dropout:
            super().__init__(*layers[:-1])
        else:
            super().__init__(*layers[:-2])


class weightConstraint(object):
    def __init__(self):
        pass
    
    def __call__(self,module):
        if hasattr(module,'weight'):
            print("Entered")
            w=module.weight.data
            w=w.clamp(-1,1)
            module.weight.data=w


def siren_init(m):
    """Initialize linear layers with SIREN-style uniform weights."""
    if isinstance(m, nn.Linear):
        with torch.no_grad():
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)


class AttentionPooling(nn.Module):
    """
    Attention Pooling layer. Takes a sequence (seq_len, batch_size, input_dim)
    and returns a fixed-length context vector (batch_size, input_dim) by learning
    attention weights over the sequence dimension.
    """
    def __init__(self, input_dim, attention_dim=None):
        super(AttentionPooling, self).__init__()
        self.input_dim = input_dim
        if attention_dim is None:
            attention_dim = max(input_dim // 2, 1)
        self.attention_layer = nn.Linear(input_dim, attention_dim)
        self.context_vector_layer = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, sequence_outputs):
        seq_len, batch_size, feat_dim = sequence_outputs.shape
        proc = sequence_outputs.permute(1, 0, 2)  # (B, S, D)
        reshaped = proc.reshape(-1, self.input_dim)
        attn_hidden = torch.tanh(self.attention_layer(reshaped))
        energy = self.context_vector_layer(attn_hidden).reshape(batch_size, seq_len)
        attention_weights = torch.nn.functional.softmax(energy, dim=1)
        context_vector = torch.sum(proc * attention_weights.unsqueeze(-1), dim=1)
        return context_vector, attention_weights
