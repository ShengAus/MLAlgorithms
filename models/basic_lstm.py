import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# From-scratch LSTM cell
class BasicLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.W = nn.Linear(input_size + hidden_size, hidden_size * 4)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h_t = torch.zeros(1, self.hidden_size)
        c_t = torch.zeros(1, self.hidden_size)
        outputs = []

        for t in range(x.size(0)):
            x_t = x[t]
            combined = torch.cat((x_t, h_t), dim=1)
            gates = self.W(combined)

            i_t, f_t, g_t, o_t = gates.chunk(4, dim=1)

            i_t = torch.sigmoid(i_t)
            f_t = torch.sigmoid(f_t)
            g_t = torch.tanh(g_t)
            o_t = torch.sigmoid(o_t)

            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)

            y_t = self.output_layer(h_t)
            outputs.append(y_t)

        return torch.stack(outputs)

