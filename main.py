import chz
from torchvision import datasets
from torchvision.transforms import v2
import torch
import numpy as np


from torch.utils.data import random_split
import PIL
def get_mnist_train_val_dl():
    train_dataset = datasets.MNIST(root='./data', train=True, download=True)
    train_split, val_split = random_split(train_dataset,[0.8,0.2], generator=torch.manual_seed(3))

    data = np.array([example[0] for example in train_split])
    train_means = data.mean()
    train_stds = data.std()

    train_dataset = datasets.MNIST(
        root='./data', train=True, download=True, transform=v2.Compose([
            v2.PILToTensor(),
            v2.ToDtype(torch.float32, scale=False),
            v2.Normalize([train_means], [train_stds])
        ])
    )

    train_split, val_split = random_split(train_dataset,[0.8,0.2], generator=torch.manual_seed(3))
    from torch.utils.data.dataloader import DataLoader

    train_dl = [data for data in DataLoader(train_split, batch_size=1, shuffle=True, num_workers=0)]
    val_dl = [data for data in DataLoader(val_split, batch_size=1, shuffle=True, num_workers=0)]
    return train_dl, val_dl

@chz.chz
class Config:
    name: str = "default"

def main(config: Config):
    print(f"Hello from {config.name}!")


if __name__ == "__main__":
    chz.entrypoint(main)
