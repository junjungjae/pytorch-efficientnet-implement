import numpy as np
from tqdm import tqdm


def calc_batch_metric(output, target):
    pred = output.argmax(1, keepdim=True)
    metric = pred.eq(target.view_as(pred)).sum().item()
    return metric


def calc_batch_loss(loss_func, output, target, opt=None):
    batch_loss = loss_func(output, target)
    batch_metric = calc_batch_metric(output, target)

    if opt is not None:
        opt.zero_grad()
        batch_loss.backward()
        opt.step()

    return batch_loss.item(), batch_metric


def loss_epoch(model, loss_func, data_loader, device, opt=None):
    epoch_loss, epoch_metric = 0, 0
    data_len = len(data_loader.dataset)

    for xd, yd in tqdm(data_loader):
        xd = xd.to(device)
        yd = yd.to(device)

        output = model(xd)

        batch_loss, batch_metric = calc_batch_loss(loss_func=loss_func, output=output, target=yd, opt=opt)

        epoch_loss += batch_loss

        if batch_metric is not None:
            epoch_metric += batch_metric

    return_loss = epoch_loss / data_len
    return_metric = epoch_metric / data_len

    return np.round(return_loss, 5), np.round(return_metric, 5)
