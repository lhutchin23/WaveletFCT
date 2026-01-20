import sys
import io


import torch
from torchinfo import summary
from model import FCT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FCT(num_classes=4, wavelet_kernel=16).to(device)

summary(model, input_size=(1, 1, 224, 224), device=device)
