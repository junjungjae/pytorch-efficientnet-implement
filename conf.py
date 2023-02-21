import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = {
    'data': {
        'dataset': '../dataset',
        'img_size': 224,
        'num_classes': 10,
        'save_weights_dir': './models',
        'device': device
    },

    'param': {
        'batch_size': 16,
        'num_epochs': 200,        
        'lr': 0.005,
    }
    
}