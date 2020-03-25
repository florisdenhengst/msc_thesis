import torch
print(torch.cuda.is_available())
print(torch.tensor([1,2,3]).to('cuda'))
