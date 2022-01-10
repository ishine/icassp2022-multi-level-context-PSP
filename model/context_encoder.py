import torch
from torch import nn
from torch.nn.functional import relu, max_pool1d


class ContextEncoder(nn.Module):
    def __init__(self, input_dim):
        super(ContextEncoder, self).__init__()
        self.kernel_size = [3, 3, 3]
        self.channels = [(input_dim, 64), (64, 32), (32, 32)]
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(ci, co, k, padding="same")
                for k, (ci, co) in zip(self.kernel_size, self.channels)
            ]
        )
        self.output_dim = sum([x[1] for x in self.channels])

    def forward(self, x):
        # x: (batch_size,context_num,x_dim)
        batch_size, context_num, _ = x.shape
        x = x.permute(0, 2, 1)
        # x: (batch_size,x_dim,context_num)
        xs = []
        for conv in self.convs:
            x = relu(conv(x))
            xs.append(x)
        # xs[0]: (batch_size,64,context_num)
        # xs[1]: (batch_size,32,context_num)
        # xs[2]: (batch_size,32,context_num)
        xs = [max_pool1d(x, x.shape[2]).squeeze(2) for x in xs]
        # xs[0]: (batch_size,64)
        # xs[1]: (batch_size,32)
        # xs[2]: (batch_size,32)
        return torch.cat(xs, dim=1).reshape((batch_size, -1))
