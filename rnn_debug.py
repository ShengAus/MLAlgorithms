# Efficient implementation equivalent to the following with bidirectional=False
import torch

# parameter initialization
batch_first = True
num_layers = 2
hidden_size = 2
input_size = 3

weight_ih = torch.randn(num_layers,hidden_size, input_size)
bias_ih = torch.randn(num_layers, hidden_size)
weight_hh = torch.randn(num_layers, hidden_size, hidden_size)
bias_hh = torch.randn(num_layers, hidden_size)


def forward(x, hx=None):
    if batch_first:
        x = x.transpose(0, 1)
    seq_len, batch_size, _ = x.size()
    if hx is None:
        hx = torch.zeros(num_layers, batch_size, hidden_size)
    h_t_minus_1 = hx
    h_t = hx
    output = []
    for t in range(seq_len):
        for layer in range(num_layers):
            # print(h_t[layer])
            # print(h_t_minus_1[layer])
            h_t[layer] = torch.tanh(
                x[t] @ weight_ih[layer].T
                + bias_ih[layer]
                + h_t_minus_1[layer] @ weight_hh[layer].T
                + bias_hh[layer]
            )
            # print(h_t[layer])
            # print(h_t_minus_1[layer])
            print(h_t[-1])
        output.append(h_t[-1].clone())
        h_t_minus_1 = h_t
        print(output)
    output = torch.stack(output)
    if batch_first:
        output = output.transpose(0, 1)
    return output, h_t

x = torch.randn(1, 2, input_size)  # (batch_size, seq_len, input_size)

result, last_hidden = forward(x)
print("Output shape:", result.shape)  # Should be (batch_size, seq_len, hidden_size)
print("Last hidden shape:", last_hidden.shape)  # Should be (num_layers, batch_size, hidden_size)