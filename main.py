'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import argparse
import os
from datetime import datetime

import torch
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import models
from utils import progress_bar

# pylint: disable=invalid-name,redefined-outer-name,global-statement

model_names = sorted(name for name in models.__dict__ if not name.startswith(
    "__") and callable(models.__dict__[name]))
best_acc = 0 # best test accuracy

parser = argparse.ArgumentParser(description='PyTorch CINIC10 Training')
parser.add_argument('data', metavar='DIR', default='data/cinic10',
                    help='path to dataset (default: data/cinic-10)')
parser.add_argument('-a', '--arch', metavar='ARCH', default='vgg16',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: vgg16)')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 64), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# Data loading code
print('==> Preparing data..')

traindir = os.path.join(args.data, 'train')
validatedir = os.path.join(args.data, 'valid')
testdir = os.path.join(args.data, 'test')
cinic_mean = [0.47889522, 0.47227842, 0.43047404]
cinic_std = [0.24205776, 0.23828046, 0.25874835]
normalize = transforms.Normalize(mean=cinic_mean, std=cinic_std)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=cinic_mean, std=cinic_std)
])

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=cinic_mean, std=cinic_std)
])

trainset = datasets.ImageFolder(root=traindir, transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=args.batch_size,
                                          shuffle=True,
                                          num_workers=args.workers)

validateset = datasets.ImageFolder(root=validatedir, transform=transform)
validateloader = torch.utils.data.DataLoader(validateset,
                                             batch_size=100,
                                             shuffle=True,
                                             num_workers=args.workers)

testset = datasets.ImageFolder(root=testdir, transform=transform)
testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=100,
                                         shuffle=True,
                                         num_workers=args.workers)

classes = ('airplane', 'automobile', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Create Model
print('==> Creating model {}...'.format(args.arch))
model = models.__dict__[args.arch]()
if args.cuda:
    # DataParallel will divide and allocate batch_size to all available GPUs
    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()

# Define loss function (criterion), optimizer and learning rate scheduler
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),
                            lr=args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)
scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=args.epochs, eta_min=0)


def train(epoch):
    ''' Trains the model on the train dataset for one entire iteration '''
    print('\nEpoch: %d' % epoch)
    cudnn.benchmark = True
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if args.cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def validate(epoch):
    ''' Validates the model's accuracy on validation dataset and saves if better
        accuracy than previously seen. '''
    cudnn.benchmark = False
    global best_acc
    model.eval()
    valid_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(validateloader):
            if args.cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            valid_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(validateloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (valid_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'model': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc


def test():
    ''' Final test of the best performing model on the testing dataset. '''
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    model.load_state_dict(checkpoint['model'])
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    print('Test best performing model from epoch {} with accuracy {:.3f}%'.format(
        checkpoint['epoch'], checkpoint['acc']))
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if args.cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))


start_time = datetime.now()
print('Runnning training and test for {} epochs'.format(args.epochs))

# Run training for specified number of epochs
for epoch in range(0, args.epochs):
    scheduler.step()
    train(epoch)
    validate(epoch)

time_elapsed = datetime.now() - start_time
print('Training time elapsed (hh:mm:ss.ms) {}\n'.format(time_elapsed))

# Run final test on never before seen data
test()
