import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = {
    'data': {
        'dataset': '../dataset',
        'img_size': 512,
        'num_classes': 10,
        'save_weights_dir': './weights',
        'device': device,
    },

    'param': {
        'batch_size': 128,
        'num_epochs': 200,        
        'lr': 0.005,
        'split_ratio': 0.9
    }
    
}