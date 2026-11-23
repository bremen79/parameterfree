# Copyright (c) 2026 Francesco Orabona
# Licensed under the MIT License. See LICENSE in the project root.

"""Compare the repository's optimizers and Adam on FashionMNIST.

Run from the repository root with:
    python -m examples.compare_fashionmnist --device cuda
"""

import argparse
import csv
import gc
import hashlib
from functools import partial
import json
import math
from pathlib import Path
import random
import time
import zipfile

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import FashionMNIST

import parameterfree
from parameterfree import COCOB, KT, cKT, ConicFTRL, CoordinateWise, BatchedOptimizer


OPTIMIZERS = {
    'Adam': torch.optim.Adam,
    'COCOB': COCOB,
    'KT': KT,
    'cKT': cKT,
    'ConicFTRL': ConicFTRL,
    'cConicFTRL': partial(CoordinateWise, ConicFTRL),
}
METRIC_FIELDS = [
    'optimizer', 'seed', 'epoch', 'online_loss', 'online_accuracy',
    'train_loss', 'train_accuracy', 'test_loss', 'test_accuracy',
    'train_seconds', 'eval_seconds', 'elapsed_seconds',
]


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


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def load_data(device, root='./data', train_limit=None, test_limit=None):
    # The original HTTP mirror can reject downloads. Use the dataset authors'
    # HTTPS mirror first; torchvision still verifies the published checksums.
    class DownloadFashionMNIST(FashionMNIST):
        mirrors = [
            'https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/',
        ] + FashionMNIST.mirrors

    data = {}
    for split, train, limit in (('train', True, train_limit), ('test', False, test_limit)):
        dataset = DownloadFashionMNIST(root=root, train=train, download=True)
        if limit is not None and not 1 <= limit <= len(dataset):
            raise ValueError('{} limit must be between 1 and {}'.format(split, len(dataset)))
        # This is the same [0, 1] scaling as ToTensor, without per-image PIL
        # conversion or per-minibatch host-to-device copies. Full data fits a T4.
        images = dataset.data[:limit].reshape(-1, 28 * 28).to(device=device, dtype=torch.float32).div_(255)
        labels = dataset.targets[:limit].to(device=device)
        data[split] = (images, labels)
    print('Loaded {} training and {} test examples on {}'.format(
        len(data['train'][1]), len(data['test'][1]), device))
    return data


@torch.no_grad()
def evaluate(model, images, labels, batch_size=1000):
    model.eval()
    metrics = torch.zeros(2, device=images.device, dtype=torch.float64)
    for start in range(0, len(labels), batch_size):
        targets = labels[start:start + batch_size]
        logits = model(images[start:start + batch_size])
        metrics[0].add_(F.cross_entropy(logits, targets, reduction='sum'))
        metrics[1].add_((logits.argmax(dim=1) == targets).sum())
    loss_sum, correct = metrics.tolist()
    if not math.isfinite(loss_sum):
        raise FloatingPointError('Non-finite evaluation loss')
    return loss_sum / len(labels), correct / len(labels)


def write_csv(path, rows, fields):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_experiment(data, epochs=30, batch_size=100, seeds=(0,), hidden_size=1000,
                   eval_batch_size=1000, cosine_decay=False,
                   output_dir='fashionmnist_results', optimizer_names=None,
                   compile=False, error_if_nonfinite=True, foreach=None):
    if min(epochs, batch_size, hidden_size, eval_batch_size) < 1 or not seeds:
        raise ValueError('Use positive epochs and batch sizes and at least one seed')
    names = list(OPTIMIZERS) if optimizer_names is None else list(optimizer_names)
    if not names or any(name not in OPTIMIZERS for name in names):
        raise ValueError('Choose optimizers from {}'.format(list(OPTIMIZERS)))
    device = data['train'][0].device
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows, statuses = [], []
    source_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(Path(parameterfree.__file__).parent.glob('*.py'))
    }
    metadata = dict(
        epochs=epochs, batch_size=batch_size, seeds=list(seeds), hidden_size=hidden_size,
        eval_batch_size=eval_batch_size, cosine_decay=cosine_decay,
        device=str(device), gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        torch_version=str(torch.__version__), optimizers=names, source_sha256=source_hashes,
        train_examples=len(data['train'][1]), test_examples=len(data['test'][1]),
        dtype='float32', input_scaling='uint8 / 255; no normalization or augmentation',
        compile=compile, error_if_nonfinite=error_if_nonfinite, foreach=foreach,
        clipping={'KT': 'global L2 norm <= 1', 'cKT': 'global Linf norm <= 1'},
        optimizer_settings={
            'Adam': dict(lr=1e-3, betas=[0.9, 0.999], eps=1e-8, weight_decay=0),
            'COCOB': dict(alpha=100, eps=1e-8, weight_decay=0),
            'KT': dict(w=1e-4, weight_decay=0, max_grad_norm=1.),
            'cKT': dict(w=1e-4, weight_decay=0, max_grad_norm=1.),
            'ConicFTRL': dict(epsilon=1, gamma=1, weight_decay=0),
            'cConicFTRL': dict(epsilon=1, gamma=1, weight_decay=0, budget='per coordinate'),
        },
    )

    def save_progress():
        write_csv(output / 'metrics.csv', rows, METRIC_FIELDS)
        (output / 'run.json').write_text(json.dumps(dict(metadata, runs=statuses), indent=2) + '\n')

    save_progress()
    train_x, train_y = data['train']
    test_x, test_y = data['test']
    steps_per_epoch = math.ceil(len(train_y) / batch_size)
    for seed in seeds:
        seed_everything(seed)
        initial = {key: value.clone() for key, value in FashionMNISTModel(hidden_size).state_dict().items()}
        for name in names:
            print('\n{} | seed {}'.format(name, seed), flush=True)
            seed_everything(seed)
            model = FashionMNISTModel(hidden_size).to(device)
            model.load_state_dict(initial)
            factory = OPTIMIZERS[name]
            optimizer_type = getattr(factory, 'func', factory)
            options = dict(compile=compile, error_if_nonfinite=error_if_nonfinite) if (
                isinstance(optimizer_type, type) and issubclass(optimizer_type, (BatchedOptimizer, COCOB))) else {}
            if name in ('COCOB', 'KT', 'cKT'):
                options['foreach'] = foreach
            if name in ('KT', 'cKT'):
                options['max_grad_norm'] = 1.0
            optimizer = factory(model.parameters(), **options)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs * steps_per_epoch) if cosine_decay else None
            # Adam has no constructor compile option. Keep its native optimizer
            # (and scheduler hooks), compiling the step callable instead.
            optimizer_step = (torch.compile(optimizer.step) if compile and name == 'Adam'
                              else optimizer.step)
            params = list(model.parameters())
            status = dict(optimizer=name, seed=seed, status='running', completed_epochs=0)
            statuses.append(status)
            save_progress()
            synchronize(device)
            run_start = time.perf_counter()
            try:
                for epoch in range(1, epochs + 1):
                    model.train()
                    # Recreate the CPU generator so every optimizer sees the same
                    # minibatch order for a given seed and epoch.
                    generator = torch.Generator().manual_seed(seed + epoch)
                    order = torch.randperm(len(train_y), generator=generator).to(device)
                    online = torch.zeros(2, device=device, dtype=torch.float64)
                    synchronize(device)
                    train_start = time.perf_counter()
                    for step, start in enumerate(range(0, len(train_y), batch_size), 1):
                        indices = order[start:start + batch_size]
                        targets = train_y[indices]
                        optimizer.zero_grad(set_to_none=True)
                        logits = model(train_x[indices])
                        loss = F.cross_entropy(logits, targets)
                        online[0].add_(loss.detach() * len(targets))
                        online[1].add_((logits.detach().argmax(dim=1) == targets).sum())
                        # Check periodically, avoiding a GPU synchronization for
                        # every minibatch solely to inspect the loss.
                        if step % 50 == 0 or step == steps_per_epoch:
                            if not math.isfinite(online[0].item()):
                                raise FloatingPointError('Non-finite training loss')
                        loss.backward()
                        optimizer_step()
                        if scheduler is not None:
                            scheduler.step()
                    synchronize(device)
                    train_seconds = time.perf_counter() - train_start
                    eval_start = time.perf_counter()
                    train_loss, train_accuracy = evaluate(model, train_x, train_y, eval_batch_size)
                    test_loss, test_accuracy = evaluate(model, test_x, test_y, eval_batch_size)
                    synchronize(device)
                    online_loss, online_correct = online.tolist()
                    row = dict(
                        optimizer=name, seed=seed, epoch=epoch,
                        online_loss=online_loss / len(train_y), online_accuracy=online_correct / len(train_y),
                        train_loss=train_loss, train_accuracy=train_accuracy,
                        test_loss=test_loss, test_accuracy=test_accuracy,
                        train_seconds=train_seconds, eval_seconds=time.perf_counter() - eval_start,
                        elapsed_seconds=time.perf_counter() - run_start,
                    )
                    rows.append(row)
                    status['completed_epochs'] = epoch
                    save_progress()
                    print('Epoch {:2d}/{:2d} | train {:.4f} | test {:.4f} | test acc {:.2%} | {:.1f}s'.format(
                        epoch, epochs, train_loss, test_loss, test_accuracy,
                        row['train_seconds'] + row['eval_seconds']), flush=True)
                status['status'] = 'completed'
            except (FloatingPointError, RuntimeError) as error:
                # Report numerical failures, retaining their partial curves.
                # Other runtime failures (including OOM) need to be fixed rather
                # than mistaken for an optimizer's numerical divergence.
                if isinstance(error, RuntimeError) and not str(error).startswith((
                        'ConicAlgorithm received non-finite', 'ConicAlgorithm scalar state lost',
                        'KTAlgorithm received non-finite')):
                    raise
                status.update(status='failed', error=str(error))
                print('FAILED: {} (partial metrics retained)'.format(error), flush=True)
            save_progress()
            del model, optimizer_step, optimizer, scheduler, params, logits, loss
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    return rows, dict(metadata, runs=statuses)


def plot_results(rows, metadata, output_dir='fashionmnist_results'):
    if not rows:
        raise ValueError('No completed epochs to plot; check the reported failures')
    colors = {'Adam': 'tab:blue', 'COCOB': 'tab:orange', 'KT': 'tab:purple',
              'cKT': 'tab:green', 'ConicFTRL': 'tab:red', 'cConicFTRL': 'tab:brown'}
    panels = [('train_loss', 'Training Loss', 'Training Loss'),
              ('test_loss', 'Testing Loss', 'Test Loss'),
              ('test_accuracy', 'Testing Accuracy', 'Test Accuracy')]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    for name in metadata['optimizers']:
        records = [row for row in rows if row['optimizer'] == name]
        failed = any(run['optimizer'] == name and run['status'] != 'completed' for run in metadata['runs'])
        label = name + (' (incomplete)' if failed else '')
        for ax, (metric, title, ylabel) in zip(axes, panels):
            epochs = sorted({row['epoch'] for row in records})
            values = [[row[metric] for row in records if row['epoch'] == epoch] for epoch in epochs]
            mean = np.array([np.mean(value) for value in values])
            std = np.array([np.std(value) for value in values])
            ax.plot(epochs, mean, label=label, color=colors[name], linewidth=2)
            if any(len(value) > 1 for value in values):
                ax.fill_between(epochs, mean - std, mean + std, color=colors[name], alpha=0.15)
            ax.set(title=title, xlabel='Epochs', ylabel=ylabel)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.grid(alpha=0.2)
            ax.legend(fontsize=10)
    title = 'FashionMNIST, cross-entropy loss, minibatch={}'.format(metadata['batch_size'])
    if metadata['train_examples'] != 60000 or metadata['test_examples'] != 10000:
        title += '\nSubset: {} train / {} test examples'.format(
            metadata['train_examples'], metadata['test_examples'])
    if len(metadata['seeds']) > 1:
        title += '\nMean ± standard deviation across available seeds'
    fig.suptitle(title, fontsize=17)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / 'fashionmnist.png', dpi=200)
    fig.savefig(output / 'fashionmnist.pdf')
    return fig


def summarize_and_archive(rows, metadata, output_dir='fashionmnist_results'):
    output = Path(output_dir)
    summary = []
    print('\nFinal-epoch results (completed runs only):')
    for name in metadata['optimizers']:
        final = [row for row in rows if row['optimizer'] == name and row['epoch'] == metadata['epochs']]
        row = dict(optimizer=name, completed_seeds=len(final), requested_seeds=len(metadata['seeds']))
        for metric in ('train_loss', 'test_loss', 'test_accuracy'):
            row[metric] = float(np.mean([value[metric] for value in final])) if final else None
        summary.append(row)
        if final:
            print('{:10} {}/{} seeds | train {:.4f} | test {:.4f} | acc {:.2%}'.format(
                name, len(final), len(metadata['seeds']), row['train_loss'], row['test_loss'], row['test_accuracy']))
        else:
            print('{}: no completed runs'.format(name))
    write_csv(output / 'summary.csv', summary,
              ['optimizer', 'completed_seeds', 'requested_seeds', 'train_loss', 'test_loss', 'test_accuracy'])
    archive = output.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(output.iterdir()):
            if path.is_file():
                handle.write(path, arcname=path.name)
    print('Saved plots, metrics, settings, and run statuses to {}'.format(archive.resolve()))
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--hidden-size', type=int, default=1000)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0])
    parser.add_argument('--train-limit', type=int)
    parser.add_argument('--test-limit', type=int)
    parser.add_argument('--cosine-decay', action='store_true')
    parser.add_argument('--compile', action='store_true')
    foreach_group = parser.add_mutually_exclusive_group()
    foreach_group.add_argument('--foreach', dest='foreach', action='store_true')
    foreach_group.add_argument('--no-foreach', dest='foreach', action='store_false')
    parser.set_defaults(foreach=None)
    parser.add_argument('--skip-finite-check', action='store_true')
    parser.add_argument('--output-dir', default='fashionmnist_results')
    args = parser.parse_args()
    data = load_data(torch.device(args.device), train_limit=args.train_limit, test_limit=args.test_limit)
    rows, metadata = run_experiment(
        data, epochs=args.epochs, batch_size=args.batch_size, hidden_size=args.hidden_size,
        seeds=args.seeds, cosine_decay=args.cosine_decay, output_dir=args.output_dir,
        compile=args.compile, error_if_nonfinite=not args.skip_finite_check, foreach=args.foreach)
    plot_results(rows, metadata, args.output_dir)
    summarize_and_archive(rows, metadata, args.output_dir)
    plt.show()


if __name__ == '__main__':
    main()
