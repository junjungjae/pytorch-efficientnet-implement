import torch
import torch.nn as nn

from models.model import EfficientNet
from datapreprocess import DataContainer
import calculate as calc
from conf import config

dc = DataContainer()
dc.run()

device = config['data']['device']
model = EfficientNet(1, 1, 1, 10)
model.load_state_dict(torch.load('./weights/best_weights.pt'))
model.to(device)

model.eval()
loss_func = nn.CrossEntropyLoss(reduction='mean')

with torch.no_grad():
    test_loss, test_metric = calc.loss_epoch(model=model, loss_func=loss_func, data_loader=dc.test_dl, device=device)

print(test_loss, test_metric*100)   