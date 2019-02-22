'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import argparse
import os
from datetime import datetime
from multiprocessing import Process

import torch
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from polyaxon_client.tracking import (Experiment, get_data_paths,
                                      get_outputs_path)
from polystores.stores.manager import StoreManager

import models
from utils import progress_bar

# pylint: disable=invalid-name,redefined-outer-name,global-statement
model_names = sorted(name for name in models.__dict__ if not name.startswith(
    "__") and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CINIC10 Training')
parser.add_argument('-a', '--arch', metavar='ARCH', default='vgg16',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: vgg16)')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--valid-batch-size', type=int, default=100, metavar='VALID',
                    help='input batch size for validating (default: 100)')
parser.add_argument('--test-batch-size', type=int, default=100, metavar='TEST',
                    help='input batch size for testing (default: 100)')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of epochs to train (default:300)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float,
                    metavar='M', help='SGD momentum factor (default: 0.9)')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W',
                    help='SGD weight decay (L2 penalty) (default: 1e-4)', dest='weight_decay')
parser.add_argument('--no-cuda', action='store_true',
                    default=False, help='disables CUDA training')
parser.add_argument('--seed', type=int, metavar='S',
                    help='set randomization seed manually')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# Polyaxon
print('Setting up Polyaxon experiment')
experiment = Experiment()
print(experiment.get_experiment_info())
print("Data paths: {}".format(get_data_paths()))
print("Outputs path: {}".format(get_outputs_path()))
experiment.set_description('PyTorch CINIC10 Benchmark')
experiment.log_params(
    model_architecture=args.arch,
    train_batch_size=args.batch_size,
    valid_batch_size=args.valid_batch_size,
    test_batch_size=args.test_batch_size,
    epochs=args.epochs,
    learning_rate=args.lr,
    data_loading_workers=args.workers,
    sgd_momentum_factor=args.momentum,
    sgd_weight_decay=args.weight_decay,
    cuda_enabled=args.cuda,
    seed=args.seed
)

store = StoreManager(path=get_data_paths()['cinic-10'])

train_dir = '/data/train'
valid_dir = '/data/valid'
test_dir = '/data/test'

# Download data via Polyaxon S3 integration
print('Starting parallel download')
start_time = datetime.now()
d1 = Process(target=store.download_dir, args=('train', train_dir))
d2 = Process(target=store.download_dir, args=('valid', valid_dir))
d3 = Process(target=store.download_dir, args=('test', test_dir))
d1.start()
d2.start()
d3.start()
print('All download threads started')
d1.join()
d2.join()
d3.join()
time_elapsed = datetime.now() - start_time
print('Download time (hh:mm:ss.ms) {}\n'.format(time_elapsed))
print('Logging data params to Polyaxon')
experiment.log_params(data_download_time=str(time_elapsed))
experiment.log_data_ref(train_dir, data_name='train')
experiment.log_data_ref(valid_dir, data_name='valid')
experiment.log_data_ref(test_dir, data_name='test')

# Data loading code
print('==> Initialize data loading..')

if args.seed:
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

cinic_mean = [0.47889522, 0.47227842, 0.43047404]
cinic_std = [0.24205776, 0.23828046, 0.25874835]
normalize = transforms.Normalize(mean=cinic_mean, std=cinic_std)
best_acc = 0  # best test accuracy

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

trainset = datasets.ImageFolder(root=train_dir, transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=args.batch_size,
                                          shuffle=True,
                                          num_workers=args.workers)

validateset = datasets.ImageFolder(root=valid_dir, transform=transform)
validateloader = torch.utils.data.DataLoader(validateset,
                                             batch_size=args.valid_batch_size,
                                             shuffle=True,
                                             num_workers=args.workers)

testset = datasets.ImageFolder(root=test_dir, transform=transform)
testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=args.test_batch_size,
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
scheduler = CosineAnnealingLR(
    optimizer=optimizer, T_max=args.epochs, eta_min=0)


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

    acc = 100.*correct/total
    experiment.log_metrics(train_accuracy=acc, epoch=epoch)


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

    acc = 100.*correct/total
    experiment.log_metrics(validate_accuracy=acc, epoch=epoch)
    # Save checkpoint.
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
    acc = 100.*correct/total
    experiment.log_metrics(test_accuracy=acc, test_epoch=checkpoint['epoch'])


start_time = datetime.now()
print('Runnning training and test for {} epochs'.format(args.epochs))

# Run training for specified number of epochs
for epoch in range(0, args.epochs):
    scheduler.step()
    train(epoch)
    validate(epoch)

time_elapsed = datetime.now() - start_time
print('Training time elapsed (hh:mm:ss.ms) {}\n'.format(time_elapsed))
experiment.log_params(training_time=str(time_elapsed))

# Run final test on never before seen data
test()

# Upload result to polyaxon output
experiment.outputs_store.upload_file('/code/checkpoint/ckpt.t7')
