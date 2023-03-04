import os
import torch

from torchvision import datasets
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms

from conf import config


class DataContainer:
    def __init__(self):
        super(DataContainer, self).__init__()
        self.datapath = config['data']['dataset']
        self.save_dir = config['data']['save_weights_dir']

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

        self.train_dl = None
        self.val_dl = None
        self.test_dl = None

    def _load_data(self):
        if not os.path.exists(self.datapath):
            os.mkdir(self.datapath)
        dataset_split_ratio = config['param']['split_ratio']
        
        # load dataset
        self.train_ds = datasets.STL10(self.datapath, split='train', download=True, transform=transforms.ToTensor())
        self.train_ds, self.val_ds = random_split(self.train_ds, [dataset_split_ratio, 1- dataset_split_ratio], torch.Generator().manual_seed(0))
        self.test_ds = datasets.STL10(self.datapath, split='test', download=True, transform=transforms.ToTensor())

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
        self.test_ds.transform = data_transform

    def _mount_dataloader(self):
        batch_size = config['param']['batch_size']
        
        self.train_dl = DataLoader(self.train_ds, batch_size=batch_size, shuffle=True)
        self.val_dl = DataLoader(self.val_ds, batch_size=batch_size, shuffle=True)
        self.test_dl = DataLoader(self.test_ds, batch_size=batch_size, shuffle=True)

    def run(self):
        self._load_data()
        self._generate_directory()
        self._transform_data()
        self._mount_dataloader()
