# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 11:13:41 2019

@author: Gentle Deng
"""

import numpy as np

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

class SiameseNet(nn.Module):
    def __init__(self, in_channel: int=3, size_kernel:int = 3, n_padding: list = [0,0,0,0],
                 n_conv: int = 4, n_featuremap: list = [16,64,128,256],
                 n_fc: int = 4, n_fc_units: list=[256, 128, 64, 1], func_act: str = 'RELU'):      
        super(SiameseNet, self).__init__()
        
        if len(n_featuremap) != n_conv or n_conv != n_padding:
            raise ValueError('Bad input! n_cov and n_padding must be equal to the length of n_featuremap!')
            
        self.n_fc   = n_fc
        self.n_conv = n_conv
        
        # add convolution layers
        self.cnn = []
        in_cnn_channel = [in_channel] + n_featuremap[:-1]
        for ct_conv in range(self.n_conv):
            # !!! it seems that we can set dilation to achieve our algrithm directly !!!
            self.cnn.append( nn.Conv2d(in_cnn_channel(ct_conv)), 
                                       n_featuremap(ct_conv), 
                                       size_kernel, 
                                       stride=1, 
                                       padding=n_padding(ct_conv)
                                       )

        # add full connected layers
        self.fc = []
        in_fc_feature = [2*n_featuremap[-1]] + n_fc_units[:-1]
        for ct_fc in range(self.n_fc):
            self.fc.append( nn.Linear(in_features=in_fc_feature(ct_fc),
                                      out_features=n_fc_units(ct_fc)) 
                                      )   
                
    def forward_cnn(self, x):
        for ct_cnn in range(self.n_conv):
            x = F.relu(self.cnn[ct_cnn](x))
        
    def forward_fc(self, x):
        for ct_fc in range(self.n_fc):
            x = F.relu(self.fc[ct_fc](x))

    def forward(self, left_patch, right_patch):
        # extract features
        left_feature = self.forward_cnn(left_patch)
        right_feature = self.forward_cnn(right_patch)
        # concatenate two feature vectors
        concat_feature = torch.cat((left_feature, right_feature), 2)
        # predict similarity
        similarity = self.forward_fc(concat_feature)
        return similarity
    
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (left_patch, right_patch, target) in enumerate(train_loader):
        left_patch, right_patch, target = left_patch.to(device), right_patch.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(left_patch, right_patch)
        loss = F.binary_cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(left_patch), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for left_patch, right_patch, target in test_loader:
            left_patch, right_patch, target = left_patch.to(device), right_patch.to(device), target.to(device)
            output = model(left_patch, right_patch)
            test_loss += F.binary_cross_entropy(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    '''
        hyperparameters of the accurate architecture are the number of convolutional layers
in each sub-network (num conv layers), the number of feature maps in each layer
(num conv feature maps), the size of the convolution kernels (conv kernel size), the size
of the input patch (input patch size), the number of units in each fully-connected layer
(num fc units), and the number of fully-connected layers (num fc layers).
    '''
    
    # Training settings
    parser = argparse.ArgumentParser(description='SiameseNet for disparity map estimation')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 10 00)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    
    parser.add_argument('--patch_size', type=list, default=[9,9], metavar='PS',
                        help='the size of input patches (default: [9,9])')
    parser.add_argument('--size_kernel', type=int, default=3, metavar='N',
                        help='the size of convolution kernel (default: 3)')
    parser.add_argument('--n_conv', type=int, default=4, metavar='N',
                        help='the number of convolution layers (default: 4)')
    parser.add_argument('--n_featuremap', type=list, default= [16,64,128,256], metavar='FM',
                        help='the number of feature maps of convolution layers (default: [16,64,128,256])')
    parser.add_argument('--n_fc', type=int, default=4, metavar='N',
                        help='the number of fully connected layers (default: 4)')
    parser.add_argument('--n_fc_units', type=list, default= [16,64,128,256], metavar='FU',
                        help='the number of units of fully connected layers (default: [16,64,128,256])')
    
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

#    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
#    
#    train_loader = torch.utils.data.DataLoader(
#        datasets.MNIST('./data', train=True, download=True,
#                       transform=transforms.Compose([
#                           transforms.ToTensor(),
#                           transforms.Normalize((0.1307,), (0.3081,))
#                       ])),
#        batch_size=args.batch_size, shuffle=True, **kwargs)
#    
#    test_loader = torch.utils.data.DataLoader(
#        datasets.MNIST('./data', train=False, transform=transforms.Compose([
#                           transforms.ToTensor(),
#                           transforms.Normalize((0.1307,), (0.3081,))
#                       ])),
#        batch_size=args.test_batch_size, shuffle=True, **kwargs)


    model = SiameseNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)

    if (args.save_model):
        torch.save(model.state_dict(),"SiameseNet.pt")
        
if __name__ == '__main__':
    main()