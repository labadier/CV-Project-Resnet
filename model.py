import numpy as np, pandas as pd, os, torch
from pathlib import Path
import cv2
from sklearn.model_selection import StratifiedKFold
import torchvision.models as models
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
from statistics import mode
from collections import OrderedDict
import json

class ResBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels, downsample):

        super(ResBlock, self).__init__()

        if downsample:
            self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                torch.nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.shortcut = torch.nn.Sequential()

        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)

    def forward(self, input):

        shortcut = self.shortcut(input)
        input = torch.nn.functional.relu(self.bn1(self.conv1(input)))
        input = torch.nn.functional.relu(self.bn2(self.conv2(input)))
        input = input + shortcut
        return torch.nn.functional.relu(input)

class ResNet30(torch.nn.Module):

  def __init__(self, in_channels=3, outputs=10):

    super(ResNet30, self).__init__()

    self.layer0 = torch.nn.Sequential(
    torch.nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
    torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
    torch.nn.BatchNorm2d(64),
    torch.nn.ReLU()
    )

    self.layer1 = torch.nn.Sequential(ResBlock(64, 64, downsample=False), ResBlock(64, 64, downsample=False),
                                      ResBlock(64, 64, downsample=False))
    self.layer2 = torch.nn.Sequential(ResBlock(64, 128, downsample=True),ResBlock(128, 128, downsample=False),
                                      ResBlock(128, 128, downsample=False), ResBlock(128, 128, downsample=False))
    self.layer3 = torch.nn.Sequential(ResBlock(128, 256, downsample=True),ResBlock(256, 256, downsample=False),
                                      ResBlock(256, 256, downsample=False), ResBlock(256, 256, downsample=False),
                                      ResBlock(256, 256, downsample=False), ResBlock(256, 256, downsample=False))
    self.layer4 = torch.nn.Sequential(ResBlock(256, 512, downsample=True),ResBlock(512, 512, downsample=False),
                                      ResBlock(512, 512, downsample=False))

    self.gap = torch.nn.AdaptiveAvgPool2d(1)
    self.fc = torch.nn.Sequential(torch.nn.Linear(512, 256), torch.nn.ReLU(), torch.nn.Linear(256, outputs))

    self.loss_criterion = torch.nn.CrossEntropyLoss() 

    self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    self.to(device=self.device)

  def forward(self, x):

    y_hat = self.layer0(x)
    y_hat = self.layer1(y_hat)
    y_hat = self.layer2(y_hat)
    y_hat = self.layer3(y_hat)
    y_hat = self.layer4(y_hat)
    y_hat = self.gap(y_hat)
    y_hat = y_hat.view(y_hat.size(0), -1)
    y_hat = self.fc(y_hat)

    return y_hat
  def load(self, path):
    self.load_state_dict(torch.load(path, map_location=self.device))

  def save(self, path):
    torch.save(self.state_dict(), path)

  def makeOptimizer(self, lr, decay=1e-4):

    # opt = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=decay)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor = 0.1, patience=5)
    opt = torch.optim.SGD(self.parameters(), lr=lr,
                      momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)
    return opt, scheduler

  def computeLoss(self, outputs, data):
    return self.loss_criterion(outputs, data['labels'].to(self.device) )


class Conv_net(torch.nn.Module):

  def __init__(self,  **kwargs):

    super(Conv_net, self).__init__()
    self.conv_chanels = [3, 32, 64, 128, 256, 512, 512, 512, 1024]
    self.dense_neurons = [512, 10]

    self.conv_layers = torch.nn.Sequential(OrderedDict([

        (f'conv_{i}',torch.nn.Sequential(torch.nn.Conv2d(in_channels=self.conv_chanels[i-1], out_channels=self.conv_chanels[i], kernel_size=(3, 3), stride=1, padding='same'),
                              torch.nn.BatchNorm2d(self.conv_chanels[i]),
                              torch.nn.LeakyReLU(0.2),
                              torch.nn.MaxPool2d(kernel_size=2),                              
                              ))
        
        for i in range(1, len(self.conv_chanels))
    ]))

    self.dense_layers = None

    self.loss_criterion = torch.nn.CrossEntropyLoss() 

    self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    self.to(device=self.device)

  def forward(self, x : torch.Tensor):
    
    y_hat = self.conv_layers(x)
    y_hat = y_hat.view(x.size(0), -1)

    if self.dense_layers is None:

      self.dense_neurons = [y_hat.size(1)] + self.dense_neurons
      self.dense_layers = torch.nn.Sequential(OrderedDict([

        (f'dense_{i}',torch.nn.Sequential(torch.nn.Linear(in_features=self.dense_neurons[i-1], out_features=self.dense_neurons[i]),torch.nn.LeakyReLU(0.2)))
        if i < len(self.dense_neurons) - 1 else \
        (f'dense_{i}',torch.nn.Linear(in_features=self.dense_neurons[i-1], out_features=self.dense_neurons[i]))

        for i in range(1, len(self.dense_neurons))
        ]))
      self.to(device=self.device)
      print('Network Created!')
      
    y_hat = self.dense_layers(y_hat)

    return  y_hat


  def load(self, path):
    self.load_state_dict(torch.load(path, map_location=self.device))

  def save(self, path):
    torch.save(self.state_dict(), path)

  def makeOptimizer(self, lr, decay=1e-4):

    # opt = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=decay)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor = 0.1, patience=5)
    opt = torch.optim.SGD(self.parameters(), lr=lr,
                      momentum=0.9, weight_decay=decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)
    return opt, scheduler

  def computeLoss(self, outputs, data):
    return self.loss_criterion(outputs, data['labels'].to(self.device) )


def train_model( trainloader, devloader, epoches, batch_size, lr, output):

  eerror, ef1, edev_error, edev_f1, eloss, dev_loss= [], [], [], [], [], []
  best_f1 = None

  model = ResNet30()
  optimizer, scheduler = model.makeOptimizer(lr=lr)

  for epoch in range(epoches):
    running_stats = {'preds': [], 'labels': [], 'loss': 0.}

    model.train()

    iter = tqdm(enumerate(trainloader, 0))
    iter.set_description(f'Epoch: {epoch:3d}')
    for j, data_batch in iter:

      torch.cuda.empty_cache()         
      inputs, labels = data_batch    
      
      optimizer.zero_grad()
      outputs = model(inputs.to('cuda'))
      loss = model.loss_criterion(outputs, labels.to('cuda'))
   
      loss.backward()
      optimizer.step()

      # print statistics
      with torch.no_grad():
        
        running_stats['preds'] += torch.max(outputs, 1)[1].detach().cpu().numpy().tolist()
        running_stats['labels'] += labels.detach().cpu().numpy().tolist()
        running_stats['loss'] += loss.item()
        
        f1 = f1_score(running_stats['labels'], running_stats['preds'], average='macro')
        error = 1. - accuracy_score(running_stats['labels'], running_stats['preds'])
        loss = running_stats['loss'] / (j+1)

      iter.set_postfix_str(f'loss:{loss:.3f} f1:{f1:.3f}, error:{error:.3f}') 

      scheduler.step()
      if j == len(trainloader) - 1:
      
        model.eval()
        eerror += [error]
        ef1 += [f1]
        eloss += [loss]

        with torch.no_grad():
          
          running_stats = {'preds': [], 'labels': [], 'loss': 0.}
          for k, data_batch_dev in enumerate(devloader, 0):
            torch.cuda.empty_cache() 

            inputs, labels = data_batch_dev    
            outputs = model(inputs.to('cuda'))

            running_stats['preds'] += torch.max(outputs, 1)[1].detach().cpu().numpy().tolist()
            running_stats['labels'] += labels.detach().cpu().numpy().tolist()

            loss = model.loss_criterion(outputs, labels.to('cuda'))
            running_stats['loss'] += loss.item()
          

          f1 = f1_score(running_stats['labels'], running_stats['preds'], average='macro')
          error = 1. - accuracy_score(running_stats['labels'], running_stats['preds'])
          loss  = running_stats['loss'] / len(devloader)
          
          edev_error += [error]
          edev_f1 += [f1]
          dev_loss += [loss]

        if best_f1 is None or best_f1 < edev_error:
          torch.save(model.state_dict(), output) 
          best_f1 = edev_error
        iter.set_postfix_str(f'loss:{eloss[-1]:.3f} f1:{ef1[-1]:.3f} error:{eerror[-1]:.3f} dev_loss: {loss:.3f} f1_dev:{f1:.3f} dev_error:{error:.3f}') 
        

  return {'loss': eerror, 'f1': ef1, 'dev_loss': edev_error, 'dev_f1': edev_f1}

    