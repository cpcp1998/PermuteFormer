import torch
from torch import nn


class PerformerAttention(nn.Module):
    def __init__(self, kernel_epsilon):
        super().__init__()
        self.kernel_epsilon = kernel_epsilon

    def forward(self, query, key, value, mask=None):
        """
        Parameters:
            query: torch.tensor(batch_size, num_heads, sequence_length, dim)
            key: torch.tensor(batch_size, num_heads, sequence_length, dim)
            value: torch.tensor(batch_size, num_heads, sequence_length, dim)
            mask: torch.tensor(batch_size, sequence_length)
        Returns:
            context: torch.tensor(batch_size, num_heads, sequence_length, dim)
        """
        query = nn.functional.relu(query) + self.kernel_epsilon
        key = nn.functional.relu(key) + self.kernel_epsilon

        if mask is not None:
            key = key * mask.unsqueeze(1).unsqueeze(-1)

        numerator = query @ (key.transpose(-2, -1) @ value)
        denominator = query @ key.transpose(-2, -1).sum(dim=-1, keepdim=True)
        output = numerator / denominator

        return output
