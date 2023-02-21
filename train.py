import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import numpy as np
import os
import time
import joblib

from models.model import EfficientNet
from datapreprocess import DataContainer
import calculate as calc
from conf import config
from mylogger import SaveLog

num_epochs = config['param']['num_epochs']
learning_rate = config['param']['lr']
save_folder = config['data']['save_weights_dir']
device = config['data']['device']

model = EfficientNet(num_classes=10, width=1, depth=1, scale=1).to(device)

dc = DataContainer()
dc.run()

loss_func = nn.CrossEntropyLoss(reduction='mean')
opt = optim.Adam(model.parameters(), lr=learning_rate)
lr_scheduler = ReduceLROnPlateau(opt)

history = {'loss': {'train': [], 'val': []}, 'metric': {'train': [], 'val': []}}

train_log = SaveLog(log_name='monitoring train', metric='accuracy')
val_log = SaveLog(log_name='monitoring validation', metric='accuracy')

best_loss = float('inf')

print(len(dc.train_dl.dataset))
print(len(dc.val_dl.dataset))

for epoch in range(1, num_epochs+1):
    start_time = time.time()
    cur_lr = opt.param_groups[0]['lr']
    print("Current Learning Rate: {}".format(cur_lr))
    
    model.train()
    train_loss, train_metric = calc.loss_epoch(model, loss_func, dc.train_dl, device, opt)
    history['loss']['train'].append(train_loss)
    history['metric']['train'].append(train_metric)

    model.eval()
    with torch.no_grad():
        val_loss, val_metric = calc.loss_epoch(model, loss_func, dc.val_dl, device)
    history['loss']['val'].append(val_loss)
    history['metric']['val'].append(val_metric)

    lr_scheduler.step(val_loss)

    print("Epoch: {} Time: {}s Train Loss: {}, Validation Loss: {}, Train Accuracy: {}%, Validation Accuracy: {}%".format(epoch, np.round(time.time() - start_time),
                              train_loss, val_loss, train_metric*100, val_metric*100))

    train_log.save(train_loss, train_metric)
    val_log.save(val_loss, val_metric)
    joblib.dump(history, './history.pkl')

    if val_loss < best_loss:
        best_loss = val_loss
        print("save best weights")
        save_dir = os.path.join(save_folder, 'weights_epoch{}.pt'.format(epoch))
        torch.save(model.state_dict(), save_dir)