# Res Experiment
from __future__ import print_function
import os
import shutil
import time
import argparse
import re
import numpy as np
import torch 
import torch.utils.data
import torchvision
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch import nn
from torch.nn import parallel
from torchvision import datasets, transforms
from torch.autograd import Variable
from load_model import FeatureExtractor
import pandas as pd

np.random.seed(100)
#model_names = sorted(name for name in models.__dict__
#    if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))


# Training Settings
parser = argparse.ArgumentParser(description='Pytorch Transfer Learning Experiment on MNIST')
parser.add_argument('--params_path', type = str, 
                    default = '/home/user/lmx/staml_final_pro/resnet50-19c8e357.pth',
                    help = 'pretrained on ImageNet params path')
parser.add_argument('--batchsize', type = int, default = 100, metavar='200', 
                    help = 'input batch size for training')
parser.add_argument('--test_batchsize', type = int, default = 100, metavar='1000',
                        help = 'input batch size for testing')
parser.add_argument('--epochs', type = int, default = 200, metavar = '10',
                    help = 'number of epochs for training')
# Currently manually lr adjustment not used
parser.add_argument('--lr', type = float, default = 0.1, metavar = 'LR',
                    help = 'Learning Rate')
parser.add_argument('--output_path', type = str, default = '/home/user/lmx/staml_final_pro/checkpoint/', metavar = 'yourpath',
                    help = 'path for storing the stat dict')
parser.add_argument('--data_dir',type=str, default = '/home/user/lmx/staml_final_pro/dataset/',
                    help='Root directory for pytorch dataloader and datasets')
parser.add_argument('-e', '--evaluate', dest='evaluate', action = 'store_true',
                    help = 'evaluate model on validation set')
parser.add_argument('--resume', default='', type=str, metavar='Stat Dict Path',
                    help = 'path to latest checkpoint')
parser.add_argument('--print_freq', type = int, default = 1, metavar = 'Print interval', 
                    help = 'Print Interval for log infomation')
parser.add_argument('-x', '--isextract', dest='isextract', action = 'store_true',
                    help = 'extract features')


best_prec1 = 0
start_epoch = 0
def main():
    global args, best_prec1, start_epoch, train_feature, val_feature
    args = parser.parse_args()
    # For output features
    train_feature = []
    val_feature = []

    #model = Wide_ResNet(params_path = args.params_path, num_classes = 10)
    model = torchvision.models.resnet50(pretrained=False)
    for param in model.parameters():
        param.requires_grad = False
    model.load_state_dict(torch.load(args.params_path))
    model.fc = nn.Sequential(
        nn.Linear(512*4, 1000),
        nn.ReLU(),
        nn.Linear(1000, 200),
        nn.ReLU(),
        nn.Linear(200, 10)
    )



    print("Pretrained ResNet50 initialization finished.")
    model.cuda()

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoints found at '{}'".format(args.resume))
    

    # The whole definition of process pipline start from here
    cudnn.benchmark = True

    # Data Loader
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5])

    # FASHION MNIST
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(root = args.data_dir, 
                        train = True, download = True, 
                        transform = transforms.Compose([transforms.Pad(98),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    normalize])),
        batch_size = args.batchsize, shuffle = True, num_workers = 2)

    val_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(root = args.data_dir, 
                        train = False, download = True,
                        transform = transforms.Compose([transforms.Pad(98),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    normalize])),
        batch_size = args.test_batchsize, shuffle = False, num_workers = 2)
    
    # For MNIST test
    #train_loader = torch.utils.data.DataLoader(
    #    datasets.MNIST(root = args.data_dir, 
    #                    train = True, download = False, 
    #                    transform = transforms.Compose([transforms.RandomResizedCrop(224),
    #                                transforms.RandomHorizontalFlip(),
    #                                transforms.ToTensor(),
    #                                normalize])),
    #    batch_size = args.batchsize, shuffle = True, num_workers = 2)

    #val_loader = torch.utils.data.DataLoader(
    #    datasets.MNIST(root = args.data_dir, 
    #                    train = False, download = False,
    #                    transform = transforms.Compose([transforms.Resize(256),
    #                                transforms.CenterCrop(224),
    #                                transforms.ToTensor(),
    #                                normalize])),
    #    batch_size = args.test_batchsize, shuffle = False, num_workers = 2
    
    if args.isextract:
        extract_feature(model, train_loader, val_loader)
        return 
    classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker Bag', 'Ankle boot')
 
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer_1 = optim.Adam(model.fc.parameters())
    #optimizer_2 = optim.SGD(model.parameters())

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(start_epoch, args.epochs):
        # No need for Adam
        # adjust_learning_rate(optimizer, epoch)
        
        # train for one epoch
        # TODO: Modify the optimizer when got stuck
        train(train_loader, model, criterion, optimizer_1, epoch)

        # evaluate
        prec1 = validate(val_loader, model, criterion)

        # save checkpoint for best prec1
        is_best = prec1 > best_prec1
        best_prec1 = prec1 if is_best else best_prec1
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': 'Res50',
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            }, is_best, args.output_path, 'checkpoint_'+str(epoch)+'.pth.tar')


def extract_feature(ext_model, train_loader, val_loader):
    feature_train = list([])
    feature_test = list([])
    exact_list = ["avgpool"]
    extractor = FeatureExtractor(ext_model, exact_list)
    #for i, (input, target) in enumerate(train_loader):
    #    input = torch.stack((input,)*3, 1).view(input.size(0), 3, input.size(2), input.size(3))
    #    input_var = Variable(input.cuda())
    #    temp_train = extractor(input_var)
    #    #print(np.shape(temp_train))
    #    feature_train.extend(temp_train)
    #    print("Train feature:%d"%i)
    #print("Train Feature extracted!",np.shape(feature_train))
    #pd.DataFrame(feature_train).to_csv('train_feature.csv',index=False)   
    for i, (input, target) in enumerate(val_loader):
        input = torch.stack((input,)*3, 1).view(input.size(0), 3, input.size(2), input.size(3))
        input_var = Variable(input.cuda())
        temp_test = extractor(input_var)
        #print(np.shape(temp_train))
        feature_test.extend(temp_test)
        print("Test feature:%d"%i)
    print("Test Feature extracted!",np.shape(feature_test))
    pd.DataFrame(feature_test).to_csv('test_feature.csv',index=False) 


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        ### For debug
        #print(input.size(), input.type())
        #input = torch.stack((input,)*3, 1).view(input.size(0), 3, input.size(2), input.size(3))
        #print("stack finish")
        #print((model.feature_extractor(input)).type(), (model.feature_extractor(input)).size() )
        #return 

        # Convert minist into 3 channels
        input = torch.stack((input,)*3, 1).view(input.size(0), 3, input.size(2), input.size(3))

        input_var = Variable(input.cuda())
        target_var = Variable(target.cuda())

        # output features
        #if epoch == 0:
        #    temp_train = 
        #    train_feature.append(input_var.data)
        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)
        losses.update(loss.data[0], input.size(0))
        # Fetch float from torch tensor
        top1.update(prec1[0][0], input.size(0))

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
        # For debug
        #return


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        input = torch.stack((input,)*3, 1).view(input.size(0), 3, input.size(2), input.size(3))
        input_var = torch.autograd.Variable(input.cuda(), volatile=True)
        target_var = torch.autograd.Variable(target.cuda(), volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0][0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1))

    print(' * Prec@1 {top1.avg:.3f} '.format(top1=top1))

    return top1.avg


def save_checkpoint(state, is_best, path, filename='checkpoint.pth.tar'):
    if is_best:
        torch.save(state, os.path.join(path, 'best_'+filename))
    #if is_best:
    #    shutil.copyfile(os.path.join(path, filename), 
    #                    os.path.join(path, 'best_'+filename))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    #print(pred,type(pred))
    #print(target,type(target))
    correct = pred.eq(target.view(1, -1).expand_as(pred).cuda())

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()