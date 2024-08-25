import torch
from torch.utils.data import DataLoader
from torchvision import datasets

def create_datasets(train_dir, test_dir, data_transform):
    train_data = datasets.ImageFolder(root=train_dir,
                                      transform=data_transform,
                                      target_transform=None)
    test_data = datasets.ImageFolder(root=test_dir,
                                     transform=data_transform)
    return train_data, test_data

def create_dataloaders(train_dataset, test_dataset, batch_size, num_workers):
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=batch_size,
                                 num_workers=num_workers,
                                 shuffle=False)
    return train_dataloader, test_dataloader

def data_setup(train_dir, test_dir, data_transform, batch_size, num_workers):
    train_dataset, test_dataset = create_datasets(train_dir=train_dir,
                                                  test_dir=test_dir,
                                                  data_transform=data_transform)
    class_names = train_dataset.classes
    train_dataloader, test_dataloader = create_dataloaders(train_dataset=train_dataset,
                                                           test_dataset=test_dataset,
                                                           batch_size=batch_size,
                                                           num_workers=num_workers)
    return train_dataloader, test_dataloader, class_names