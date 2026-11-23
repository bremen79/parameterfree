# Copyright (c) 2023, 2024, 2026 Francesco Orabona
# Licensed under the MIT License. See LICENSE in the project root.

"""Train one optimizer on FashionMNIST using the current PyTorch interface.

From the repository root:
    python -m examples.test_on_fashionmnist --optimizer COCOB
    python -m examples.test_on_fashionmnist --optimizer cKT --compile

Requires torchvision and numpy in addition to the library's torch dependency.
"""

import argparse
from functools import partial
import math
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import FashionMNIST as _FashionMNIST
from torchvision.transforms import ToTensor

from parameterfree import COCOB, KT, cKT, ConicFTRL, CoordinateWise


OPTIMIZERS = {
    'COCOB': COCOB,
    'KT': KT,
    'cKT': cKT,
    'ConicFTRL': ConicFTRL,
    'cConicFTRL': partial(CoordinateWise, ConicFTRL),
    'Adam': torch.optim.Adam,
}


class FashionMNIST(_FashionMNIST):
    # Keep the original class name so existing FashionMNIST/raw caches work.
    # torchvision verifies the authors' downloads using the standard checksums.
    mirrors = ['https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/'] + _FashionMNIST.mirrors


class FashionMNISTModel(nn.Module):
    def __init__(self, hidden_size=1000):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 10)

    def forward(self, x):
        x = x.flatten(1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def make_optimizer(name, params, compile=False, error_if_nonfinite=True,
                   foreach=None, cosine_decay=False):
    params = list(params)
    if compile and not hasattr(torch, 'compile'):
        raise ValueError('--compile requires PyTorch 2 or newer')
    options = {}
    if name == 'Adam':
        if foreach is not None:
            options['foreach'] = foreach
        if compile and cosine_decay:
            # Tensor lr avoids recompiling Adam for each scheduler value.
            device = params[0].device
            if device.type == 'cpu' and foreach:
                raise ValueError('Compiled Adam cosine scheduling on CPU requires --no-foreach')
            options['lr'] = torch.tensor(1e-3, device=device, dtype=params[0].dtype)
            options['capturable'] = device.type == 'cuda'
    else:
        options.update(compile=compile, error_if_nonfinite=error_if_nonfinite)
        if name in ('COCOB', 'KT', 'cKT'):
            options['foreach'] = foreach
        if name in ('KT', 'cKT'):
            # Required bounds: global L2 for KT, global Linf for cKT.
            # No weight decay or gradient multiplier above 1 is used here.
            options['max_grad_norm'] = 1.0
    return OPTIMIZERS[name](params, **options)


def _metrics(totals, count):
    if count == 0:
        raise ValueError('Cannot evaluate an empty data loader')
    loss_sum, correct = totals.tolist()
    if not math.isfinite(loss_sum):
        raise FloatingPointError('Non-finite loss')
    return loss_sum / count, correct / count


def train_epoch(model, loader, optimizer, device, *, update=None, scheduler=None,
                epoch=1, log_interval=100):
    model.train()
    totals = torch.zeros(2, dtype=torch.float64, device=device)
    count = 0
    update = optimizer.step if update is None else update
    for batch_index, (images, targets) in enumerate(loader, 1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        update()
        if scheduler is not None:
            scheduler.step()
        # Sample-weighted, per-epoch online metrics. Keep accumulation on the
        # device rather than synchronizing CUDA for every minibatch.
        totals[0].add_(loss.detach() * len(targets))
        totals[1].add_((logits.detach().argmax(dim=1) == targets).sum())
        count += len(targets)
        if log_interval and (batch_index % log_interval == 0 or batch_index == len(loader)):
            average_loss, accuracy = _metrics(totals, count)
            print('Epoch {} | batch {}/{} | online loss {:.4f} | accuracy {:.2%}'.format(
                epoch, batch_index, len(loader), average_loss, accuracy), flush=True)
    return _metrics(totals, count)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    totals = torch.zeros(2, dtype=torch.float64, device=device)
    count = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        totals[0].add_(F.cross_entropy(logits, targets, reduction='sum'))
        totals[1].add_((logits.argmax(dim=1) == targets).sum())
        count += len(targets)
    return _metrics(totals, count)


def _limit(dataset, count):
    if count is None:
        return dataset
    if not 1 <= count <= len(dataset):
        raise ValueError('Dataset limits must be between 1 and {}'.format(len(dataset)))
    return Subset(dataset, range(count))


def _positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--optimizer', choices=list(OPTIMIZERS), default='COCOB')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--data-dir', default='./data')
    parser.add_argument('--epochs', type=_positive_int, default=30)
    parser.add_argument('--batch-size', type=_positive_int, default=100)
    parser.add_argument('--hidden-size', type=_positive_int, default=1000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--log-interval', type=int, default=100, help='batches between logs; 0 logs only epoch results')
    parser.add_argument('--train-limit', type=_positive_int)
    parser.add_argument('--test-limit', type=_positive_int)
    parser.add_argument('--compile', action='store_true', help='compile optimizer updates; first epoch includes compilation')
    parser.add_argument('--skip-finite-check', action='store_true', help='omit repository optimizers\' per-step validation')
    parser.add_argument('--cosine-decay', action='store_true', help='decay lr/gradient multiplier over all training steps')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--foreach', dest='foreach', action='store_true', help='multi-tensor Adam/COCOB/KT/cKT updates')
    group.add_argument('--no-foreach', dest='foreach', action='store_false')
    parser.set_defaults(foreach=None)
    args = parser.parse_args(argv)
    if not 0 <= args.seed < 2 ** 32:
        parser.error('--seed must be between 0 and 4294967295')
    if args.workers < 0 or args.log_interval < 0:
        parser.error('--workers and --log-interval must be nonnegative')
    device = torch.device(args.device)
    if device.type not in ('cpu', 'cuda'):
        parser.error('This example supports CPU and CUDA devices')
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable; use --device cpu')
    if args.compile and not hasattr(torch, 'compile'):
        parser.error('--compile requires PyTorch 2 or newer')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    train_data = _limit(FashionMNIST(args.data_dir, train=True, transform=ToTensor(), download=True), args.train_limit)
    test_data = _limit(FashionMNIST(args.data_dir, train=False, transform=ToTensor(), download=True), args.test_limit)
    options = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=device.type == 'cuda')
    # Evaluation has a separate sequential loader: it never consumes the
    # training sampler's shuffle generator.
    train_loader = DataLoader(train_data, shuffle=True, generator=torch.Generator().manual_seed(args.seed), **options)
    train_eval_loader = DataLoader(train_data, shuffle=False, **options)
    test_loader = DataLoader(test_data, shuffle=False, **options)
    model = FashionMNISTModel(args.hidden_size).to(device=device, dtype=torch.float32)
    optimizer = make_optimizer(args.optimizer, model.parameters(), compile=args.compile,
                               error_if_nonfinite=not args.skip_finite_check,
                               foreach=args.foreach, cosine_decay=args.cosine_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(train_loader)) if args.cosine_decay else None
    # Create this callable after the scheduler installs its optimizer hook.
    update = torch.compile(optimizer.step) if args.compile and args.optimizer == 'Adam' else optimizer.step
    print('{} on {} | seed {} | {} train / {} test examples'.format(
        args.optimizer, device, args.seed, len(train_data), len(test_data)), flush=True)
    for epoch in range(1, args.epochs + 1):
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        train_epoch(model, train_loader, optimizer, device, update=update,
                    scheduler=scheduler, epoch=epoch, log_interval=args.log_interval)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        train_seconds = time.perf_counter() - start
        train_loss, train_accuracy = evaluate(model, train_eval_loader, device)
        test_loss, test_accuracy = evaluate(model, test_loader, device)
        print('Epoch {}/{} | train loss {:.4f}, accuracy {:.2%} | test loss {:.4f}, accuracy {:.2%} | training {:.2f}s'.format(
            epoch, args.epochs, train_loss, train_accuracy, test_loss, test_accuracy, train_seconds), flush=True)


if __name__ == '__main__':
    main()
