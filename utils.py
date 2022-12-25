
from torchvision import transforms, datasets
from torch.utils.data import  DataLoader
import numpy as np
from matplotlib import pyplot as plt
import os

class params:
  
    output = 'out'
    ep = 120
    bs = 400
    lr = 0.01


def load_dataset(batch_size):

  transform = transforms.Compose(
      [transforms.ToTensor(),
       transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
      ]
      )

  transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010), inplace=True)
      ])

  train_loader = DataLoader(datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train),
                                            batch_size=batch_size,
                                            shuffle=True)

  dev_loader = DataLoader(datasets.CIFAR10(root='./data', train=False, download=True, transform=transform),
                                          batch_size=batch_size,
                                          shuffle=False)
  
  return train_loader, dev_loader



def plot_training(history, model, output, measure='loss'):
    
    plt.plot(history[measure])
    plt.plot(history['dev_' + measure])
    plt.legend(['train', 'dev'], loc='upper left')
    plt.ylabel(measure)
    plt.xlabel('Epoch')
    if measure == 'loss':
        x = np.argmin(history['dev_loss'])
    else: x = np.argmax(history['dev_f1'])

    plt.plot(x,history['dev_' + measure][x], marker="o", color="red")
    plt.savefig(os.path.join(output, f'train_history_{model}.png'))