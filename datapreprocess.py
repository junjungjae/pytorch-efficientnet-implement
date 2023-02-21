from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import os
from conf import config


class DataContainer:
    """
    torchvision에 내장된 데이터셋을 불러와 작업을 수행하는 방식
    """

    def __init__(self):
        super(DataContainer, self).__init__()
        self.datapath = config['data']['dataset']
        self.save_dir = config['data']['save_weights_dir']

        self.train_ds = None
        self.val_ds = None

        self.train_dl = None
        self.val_dl = None

    def _load_data(self):
        if not os.path.exists(self.datapath):
            os.mkdir(self.datapath)

        # load dataset
        self.train_ds = datasets.STL10(self.datapath, split='train', download=False, transform=transforms.ToTensor())
        self.val_ds = datasets.STL10(self.datapath, split='test', download=False, transform=transforms.ToTensor())

    def _generate_directory(self):
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

    def _transform_data(self):
        img_size = config['data']['img_size']

        data_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.train_ds.transform = data_transform
        self.val_ds.transform = data_transform

    def _mount_dataloader(self):
        batch_size = config['param']['batch_size']
        self.train_dl = DataLoader(self.train_ds, batch_size=batch_size, shuffle=True)
        self.val_dl = DataLoader(self.val_ds, batch_size=batch_size, shuffle=True)

    def run(self):
        self._load_data()
        self._generate_directory()
        self._transform_data()
        self._mount_dataloader()
