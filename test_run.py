import torch
import torch.nn as nn

rel_func = nn.ReLU()
input = torch.randn(2)
output = rel_func(input)

torch.seed()

tensor_sample = torch.tensor(17)


try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_name = torch.cuda.get_device_name(device)
    print(device_name)
    print(device)

except Exception as e:
    print(e)

print(tensor_sample)
print(output)

