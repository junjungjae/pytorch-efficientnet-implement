import torch
import os

import numpy as np
from conf import config

class EarlyStopping:
    def __init__(self, mode, patience=10, verbose=False, delta=0):
        self.mode = mode
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.weights_save_dir = config['data']['save_weights_dir']
        
        if self.mode == 'metric':
            self.metric_min = 0
        
        else:
            self.metric_min = np.inf
        
        print(f"EarlyStopping mode: {self.mode}\nPatience: {self.patience}")

    def __call__(self, target_score, model):
        if self.mode == 'metric':
            score = target_score
        
        elif self.mode == 'loss':
            score = -target_score
        
        else:
            raise ValueError("mode param must be selected 'metric' or 'loss'.")
        
        self.save_checkpoint(target_score, model, islast=True)

        if not self.best_score:
            self.save_checkpoint(target_score, model)
            self.best_score = target_score
            
        elif (score < self.best_score + self.delta):
            print(f"Current score: {score}, best score: {self.best_score}")
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                
        else:
            self.save_checkpoint(target_score, model)
            self.counter = 0

    def save_checkpoint(self, target_score, model, islast=False):        
        
        if not os.path.isdir(self.weights_save_dir):
            os.mkdir(self.weights_save_dir)
        
        if islast:
            torch.save(model.state_dict(), f"{self.weights_save_dir}/last_weights.pt")
            
        else:
            if self.verbose and self.best_score:
                print(f'Validation {self.mode} improved ({self.best_score:.6f} --> {target_score:.6f}).  Saving model ...')
                torch.save(model.state_dict(), f"{self.weights_save_dir}/best_weights.pt")
                self.best_score = target_score
            
            else:
                torch.save(model.state_dict(), f"{self.weights_save_dir}/best_weights.pt")
                self.best_score = target_score