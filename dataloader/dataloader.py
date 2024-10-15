# dataloader/dataloader.py

import os
import sys
import requests
from collections import namedtuple
from contextlib import contextmanager
from .preprocessors import default_preprocess
from .utils import download_file, timer

DataSample = namedtuple('DataSample', ['features', 'label'])

KnownDatasetPaths = {'MNIST': 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz',
                     'CIFAR-10': 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
                     'CIFAR-100': 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'}    

class DataLoader:
    def __init__(self, dataset_name='MNIST', batch_size=32, shuffle=True, **kwargs):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.shuffle = shuffle        
        self.data = []
        self.index = 0
        self.path = None
        self.kwargs = kwargs
        for key, value in kwargs.items():
            if key == 'path':
                self.path = value
                break
        if (self.path is None) and (self.dataset_name in KnownDatasetPaths):
            self.path = KnownDatasetPaths[self.dataset_name] 
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}, please specify a path in kwargs")
        
        self.load_data()
    
    @timer
    def load_data(self):
        # Get the parent directory of the current file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        self.dataset_path = os.path.join(parent_dir, 'datasets', self.dataset_name)
        if not os.path.exists(self.dataset_path):
            self.download_dataset(self.dataset_path)
        # Implement data loading logic
        (x_train, y_train), (x_test, y_test) = self.read_data()
        self.data =  (self.preprocess_data(x_train), y_train)
        self.test_data = (self.preprocess_data(x_test), y_test)
    
    @timer
    def download_dataset(self, dataset_dir):
        # Implement dataset download logic
        #Create folder with self.dataset_name
        os.makedirs(dataset_dir, exist_ok=True)
        os.chmod(dataset_dir, 0o777)  # Set full read, write, and execute permissions
        print(f"Downloading {self.dataset_name} dataset...")        
        # Use download_file from utils.py
        download_file(self.path, dataset_dir)
    
    def read_data(self):
        # Implement data reading logic
        import numpy as np
        x_train = None
        y_train = None
        x_test = None
        y_test = None
        if self.dataset_name == 'MNIST':
            x_train = np.load(self.dataset_path + '/x_train.npy')
            y_train = np.load(self.dataset_path + '/y_train.npy')
            x_test = np.load(self.dataset_path + '/x_test.npy')
            y_test = np.load(self.dataset_path + '/y_test.npy')
        elif self.dataset_name == 'CIFAR10':
            x_train = np.load('path_to_npy/x_train.npy')
            y_train = np.load('path_to_npy/y_train.npy')
        elif self.dataset_name == 'CIFAR100':
            x_train = np.load('path_to_npy/x_train.npy')
            y_train = np.load('path_to_npy/y_train.npy')            
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
        return (x_train, y_train), (x_test, y_test)
    
    def preprocess_data(self, data):
        # Implement data preprocessing logic
        preprocess_func = self.kwargs.get('preprocess_func', default_preprocess)
        return [preprocess_func(sample) for sample in data]
    
    def __iter__(self):
        self.index = 0
        if self.shuffle:
            import random
            random.shuffle(self.data)
        return self
    
    def __next__(self):
        if self.index < len(self.data):
            batch = self.data[self.index:self.index + self.batch_size]
            self.index += self.batch_size
            return batch
        else:
            raise StopIteration