# src/data/load_data.py

import torch
from torchvision import datasets, transforms

def get_dataloaders(processed_dir, batch_size=32, img_size=224, num_workers=2):  
    # Common transform for all splits
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # same normalization as preprocessing
    ])
    
    dataloaders = {}
    splits = ['train', 'val', 'test']
    
    for split in splits:
        dataset = datasets.ImageFolder(root=f"{processed_dir}/{split}", transform=transform)
        shuffle = True if split == 'train' else False
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )
        dataloaders[split] = loader
    
    return dataloaders
