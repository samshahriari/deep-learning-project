import torch
from torch import nn

class RNN(nn.Module):
    def __init__(self, num_unique_chars, hidden_size, output_size, embedding_size=None, embedding_weights=None):
        super(RNN, self).__init__()
        if embedding_size is None:
            embedding_size = hidden_size
        self.embed = nn.Embedding(num_unique_chars, embedding_size)
        if embedding_weights is not None:
            print("using pretrained embeddings")
            self.embed.weight = nn.Parameter( torch.tensor(embedding_weights, dtype=torch.float), requires_grad=True)
        self.RNN = nn.RNN(embedding_size, hidden_size, batch_first=True)
        self.final = nn.Linear(hidden_size, output_size)

    def forward(self, x, h0=None):
        x = self.embed(x)
        output, last_hidden_state = self.RNN(x, h0)
        # print(last_hidden_state.shape)
        return self.final(output), last_hidden_state.detach()


class LSTM(nn.Module):
    def __init__(self, num_unique_chars, hidden_size, output_size, num_layers, embedding_size=None, embedding_weights=None):
        super(LSTM, self).__init__()
        if embedding_size is None:
            embedding_size = hidden_size
        self.embed  = nn.Embedding(num_unique_chars, embedding_size)
        if embedding_weights is not None:
            print("using pretrained embeddings")
            self.embed.weight = nn.Parameter( torch.tensor(embedding_weights, dtype=torch.float), requires_grad=True)
        self.LSTM = nn.LSTM(embedding_size, hidden_size, num_layers, batch_first=True)
        self.final = nn.Linear(hidden_size, output_size)

    def forward(self, x, h0=None):
        x = self.embed(x)
        output, last_hidden_state = self.LSTM(x, h0)
        return self.final(output), [h.detach() for h in last_hidden_state]
    
