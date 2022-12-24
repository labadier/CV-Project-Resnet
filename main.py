import argparse, sys, os, numpy as np, torch, random
from pathlib import Path
from utils import load_dataset, plot_training, params

from model import train_model

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

def check_params(args=None):
  parser = argparse.ArgumentParser(description='Deforestation Detector')

  parser.add_argument('-output', metavar='output', default = params.output, help='output directory')
  parser.add_argument('-ep', metavar='ep', type=int, default = params.ep, help='Epoches to train')
  parser.add_argument('-bs', metavar='bs', type=int, default = params.bs, help='Batch Size')
  parser.add_argument('-lr', metavar='lr', type=float, default=params.lr, help='Learning Rate')

  return parser.parse_args(args)

if __name__ == '__main__':


  parameters = check_params(sys.argv[1:])

  output = parameters.output
  ep = parameters.ep
  bs = parameters.bs
  lr = parameters.lr
  

  Path(output).mkdir(parents=True, exist_ok=True)
  trainloader, devloader = load_dataset(bs)

  history = train_model(trainloader, devloader, epoches = ep, batch_size = bs, lr = lr, output=os.path.join(output, 'best_model'))
  plot_training(history, output, 'loss')