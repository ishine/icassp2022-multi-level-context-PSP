import torch
from torch import nn
from torch.nn.functional import relu, max_pool1d, avg_pool1d


class SentenceEncoder(nn.Module):
    def __init__(self, input_dim):
        super(SentenceEncoder, self).__init__()
        self.kernel_size = [3, 3, 3]
        self.channels = [(input_dim, 128), (128, 64), (64, 64)]
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(ci, co, k, padding="same")
                for k, (ci, co) in zip(self.kernel_size, self.channels)
            ]
        )
        self.output_dim = sum([x[1] for x in self.channels])

    def forward(self, x):
        # x: (batch_size,context_num,max_seq_len,x_dim)
        batch_size, context_num, max_seq_len, x_dim = x.shape
        x = x.reshape((batch_size * context_num, max_seq_len, -1)).permute(0, 2, 1)
        # x: (batch_size*context_num,x_dim,max_seq_len)
        xs = []
        for conv in self.convs:
            x = relu(conv(x))
            xs.append(x)
        # xs[0]: (batch_size*context_num,128,max_seq_len)
        # xs[1]: (batch_size*context_num,64,max_seq_len)
        # xs[2]: (batch_size*context_num,64,max_seq_len)
        xs = [max_pool1d(x, x.shape[2]).squeeze(2) for x in xs]
        # xs[0]: (batch_size*context_num,128)
        # xs[1]: (batch_size*context_num,64)
        # xs[2]: (batch_size*context_num,64)
        return torch.cat(xs, dim=1).reshape((batch_size, context_num, -1))
