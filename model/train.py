#!/usr/bin/env python
"""
Tools for training a model on the UCI Energy Efficiency Dataset 2012.
"""

import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

from model.dataset import EnergyEfficiencyDataset
from model.model import Model


def train(net: torch.nn.Module,
          device: torch.device,
          loader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer):
    """
    Perform one epoch of training.
    """
    net.train()
    for _, (features, targets) in enumerate(loader):
        features, targets = features.to(device), targets.to(device)
        optimizer.zero_grad()
        output = net(features)
        loss = F.mse_loss(output, targets)
        loss.backward()
        optimizer.step()


def test(net: torch.nn.Module,
         device: torch.device,
         loader: torch.utils.data.DataLoader,
         epoch: int) -> float:
    """
    Test the model.
    """
    net.eval()
    loss = 0
    with torch.no_grad():
        for features, targets in loader:
            features, targets = features.to(device), targets.to(device)
            output = net(features)
            loss += F.mse_loss(output, targets, reduction='sum').item()

    loss /= len(loader.dataset)

    print(f"Epoch {epoch} - Loss: {loss}")

    return loss


def run_training():
    """
    Train a model on the UCI Energy Efficiency Dataset 2012.
    """
    parser = argparse.ArgumentParser(
        description="Train a model on the UCI Energy Efficiency Dataset 2012.")
    parser.add_argument('--batch-size', type=int, default=100,
                        help="batch size on which to train the model")
    parser.add_argument('--data_dir', type=str, default='data',
                        help="directory in which to store the data")
    parser.add_argument('--epochs', type=int, default=200,
                        help="how many epochs to train the model")
    parser.add_argument('--early_stopping', type=int, default=5,
                        help=("how many epochs with no improvements to wait "
                              "before stopping"))
    parser.add_argument('--lr', type=float, default=0.001,
                        help="learning rate with which to train the model")
    parser.add_argument('--model_dir', type=str, default='models',
                        help="directory in which to store the saved model")
    parser.add_argument('--cuda', action='store_true', default=False,
                        help="enable training on the GPU")
    args = parser.parse_args()

    # CUDA is disabled by default because the training data is small enough
    # that it is faster to run on the CPU.
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        EnergyEfficiencyDataset(args.data_dir, train=True),
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs)
    test_loader = torch.utils.data.DataLoader(
        EnergyEfficiencyDataset(args.data_dir, train=False),
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs)

    net = Model().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    # Train model
    waited_epochs = 0
    best_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        train(net, device, train_loader, optimizer)
        loss = test(net, device, test_loader, epoch)
        if loss >= best_loss:
            waited_epochs += 1
            if waited_epochs > args.early_stopping:
                break
        else:
            best_loss = loss
            waited_epochs = 0

    # Save trained model
    save_path = (Path(args.model_dir)
                 / (datetime.now().strftime("%Y%m%d-%H%M%S") + '.pt'))
    print(f"Saving to {save_path}")
    save_path.parent.mkdir(exist_ok=True)
    torch.save(net.state_dict(), save_path)
