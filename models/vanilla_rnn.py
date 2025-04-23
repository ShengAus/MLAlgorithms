import torch
import torch.nn as nn
import torch.nn.functional as F

class VanillaRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(VanillaRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Learnable weights for input-to-hidden and hidden-to-hidden
        self.W_ih = nn.Parameter(torch.randn(hidden_size, input_size))
        self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.b_h = nn.Parameter(torch.zeros(hidden_size))

        # Output layer
        self.W_ho = nn.Linear(hidden_size, output_size)

    def forward(self, x_seq, h0=None):
        """
        x_seq: tensor of shape (seq_len, input_size)
        h0: optional initial hidden state (hidden_size,)
        """
        seq_len, _ = x_seq.size()
        h = h0 if h0 is not None else torch.zeros(self.hidden_size)
        outputs = []

        for t in range(seq_len):
            x_t = x_seq[t]
            h = torch.tanh(self.W_ih @ x_t + self.W_hh @ h + self.b_h)  # RNN recurrence
            y_t = self.W_ho(h)  # optional output layer
            outputs.append(y_t)

        return torch.stack(outputs), h
