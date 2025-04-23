import torch

# x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)

# row = x[0]  # now row is a copy
# row[0] = 999

# print(x)            # changed!


# Example 1, they points to the same memeory
h_t = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)

h_t_minus_1 = h_t  # now h_t_minus_1 is a view

h_t[0] = torch.tan(h_t_minus_1[0] + 1)

print(h_t)         
print(h_t_minus_1)


# Example 2
output = []
h_t = [torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)]
output.append(h_t[-1])  # now output[0] is a view
h_t[-1] = torch.tanh(h_t[-1] + 1)
output.append(h_t[-1])  # now output[1] is a view
print(output)

