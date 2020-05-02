from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from capcode import CapchaDataset
from PIL import Image
import numpy as np
from torchvision.utils import *

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5,stride=(1,1))  
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        #self.conv25=nn.Conv2d(20,20, kernel_size=1)
        self.conv3=nn.Conv2d(20,64, kernel_size=5)

        self.conv4=nn.Conv2d(64,2,kernel_size=1)

        

    def forward(self, x):
        
        x = F.relu(F.max_pool2d(self.conv1(x), 2))    #32->28->14
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))  #14->10->5

        x = F.dropout(x, training=self.training)
        x=torch.sigmoid(self.conv3(x) ) # 5->1
        x=self.conv4(x)  #.mean(dim=3)
        x=F.softmax(x,dim=1).max(dim=3)[0]
        x=x.max(dim=2)[0]
        x=x.view(-1,2)
        #print(x.size())  #64,2,1
        
        return x  #F.log_softmax(x, dim=1)

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        target=target.view(-1)
        
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)[:,0]
        
        loss = F.binary_cross_entropy(output, target)#(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            target=target.view(-1)
            
            data, target = data.to(device), target.to(device)
            output = model(data)[:,0]
            #test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            test_loss += F.binary_cross_entropy(output, target).item()
            #pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            pred=(output/0.5).int()
            correct += pred.eq(target.int()).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    print('my device is :',device)
    kwargs = {'num_workers': 12, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        CapchaDataset( 
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        CapchaDataset( 
                        transform=transforms.Compose([
                           transforms.ToTensor(),
                       ])),
        batch_size=args.test_batch_size, shuffle=False, **kwargs)


    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)

def train_model(model=None):
    args=None
    use_cuda = True
    lr=0.01
    epoch=20
    torch.manual_seed(10)

    device = torch.device("cuda" if use_cuda else "cpu")
    print('my device is :',device)
    kwargs = {'num_workers': 16, 'pin_memory': True} if use_cuda else {}
    
    test_loader = torch.utils.data.DataLoader(
        CapchaDataset(  ratio=1,
                        transform=transforms.Compose([
                           transforms.ToTensor(),
                       ])),
        batch_size=1000, shuffle=False, **kwargs)


    model = model if model else Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr )

    for epoch in range(1, epoch + 1):
        train_loader = torch.utils.data.DataLoader(
        CapchaDataset( ratio=3,#20-epoch,
                        txtnum=2,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                       ])),
        batch_size=64, shuffle=False, **kwargs)
        train(args, model, device, train_loader, optimizer, epoch)
        if epoch%20==0:
            test(args, model, device, test_loader)
    return model

def heatmap(md,txtnum=4):
    kwargs = {'num_workers': 1, 'pin_memory': True} 
    data_loader = torch.utils.data.DataLoader(
        CapchaDataset( txtnum=txtnum,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                       ])),
        batch_size=64, shuffle=False, **kwargs)
    md.eval()
    d={}
    def for_hook(md,input,output):
        d[0]=output.detach()
    hk=md.conv4.register_forward_hook(for_hook)
    for data,label in data_loader:
        output=md(data.to('cuda'))
        break
    r=F.softmax(d[0],dim=1)
    print(r[:,0,:,:].shape)
    #imgs=[Image.fromarray(r[i,0,:,:].cpu().numpy()*255).resize((136,32),1)  for i in range(64)]
    imgs=[Image.fromarray(r[i,0,:,:].cpu().numpy()*255).resize((136,136),1)  for i in range(64)]
    imar=np.array([[np.array(img)/255.0] for img in imgs])
    
    imd=torch.from_numpy(imar)
    save_image(imd,'pr.png')
    save_image(data,'data.png')
    print('done.....')
if __name__ == '__main__':
    md=train_model()
    heatmap(md)
    
