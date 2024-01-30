import argparse

from utils import *
from functions import *
from data import *
from models import ThreeLayerViH
import matplotlib.pyplot as plt
import random
import numpy as np
import seaborn as sns
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from einops import rearrange, repeat

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--data', type=str, default='tiny_imagenet')
    parser.add_argument('--beta', default=None)
    parser.add_argument('--update_steps', type=int, default=1)
    parser.add_argument('--kernel_epoch', type=int, default=20)
    parser.add_argument('--activation', type=str, default='softmax1')
    parser.add_argument('--mode', type=str, default='UMHN')
    parser.add_argument('--kernel', type=str, default='lin')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--datasize', type=int, default=1000)
    parser.add_argument('--patch_size', type=int, default=32)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--init_lr', type=float, default=1e-3)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--n_class', type=int, default=200)
    parser.add_argument('--channel', type=int, default=3)
    parser.add_argument('--n_heads', type=int, default=4)

    args = parser.parse_args()
    return args

class Exp:
    def __init__(self, args):
        self.args = args
        torch.manual_seed(self.args.seed)

        self.cri = nn.CrossEntropyLoss()
        self.model = ThreeLayerViH(in_channels=self.args.channel, patch_size=args.patch_size, emb_size=args.d_model, img_size=args.img_size*args.img_size, n_heads=args.n_heads, mode=args.mode, n_class=args.n_class).cuda() # embedding size P^2*C
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.init_lr)

    def get_loaders(self):

        if self.args.data == 'cifar10':
            # norm args are for cifar10
            train_transform = transforms.Compose([transforms.ToTensor(), 
                                        transforms.Normalize(mean=[0.4914,0.4822,0.4465],std=[0.2470,0.2435,0.2616])])
                                        #   transforms.RandomHorizontalFlip(0.5), 
                                        #   transforms.RandomCrop(size=[32,32], padding=4)])
            test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.4942,0.4851,0.4504],std=[0.2467,0.2429,0.2616])])
            train_data = datasets.CIFAR10(root='./cifar_data', train=True, download=True, transform=train_transform)
            test_data = datasets.CIFAR10(root='./cifar_data', train=False, download=True, transform=test_transform)
            train_indices = random.sample(range(len(train_data)), self.args.datasize)
        elif self.args.data == 'mnist':
            train_transform = transforms.Compose([transforms.ToTensor(), 
                                        transforms.Normalize((0.1307,), (0.3081,)),
                                          transforms.RandomHorizontalFlip(0.5), 
                                          transforms.RandomCrop(size=[64,64], padding=4)])
            test_transform = train_transform
            train_data = datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
            test_data = datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)
            train_indices = random.sample(range(len(train_data)), self.args.datasize)

        elif self.args.data == 'cifar100':
            train_transform = transforms.Compose([transforms.ToTensor(), 
                                        transforms.Normalize(mean=[0.4914,0.4822,0.4465],std=[0.2470,0.2435,0.2616])])
                                        #   transforms.RandomHorizontalFlip(0.5), 
                                        #   transforms.RandomCrop(size=[32,32], padding=4)])
            test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.4942,0.4851,0.4504],std=[0.2467,0.2429,0.2616])])
            train_data = datasets.CIFAR100(root='./cifar_data', train=True, download=True, transform=train_transform)
            test_data = datasets.CIFAR100(root='./cifar_data', train=False, download=True, transform=test_transform)
            train_indices = random.sample(range(len(train_data)), self.args.datasize)

        elif self.args.data == 'tiny_imagenet':
            train_data, train_label, test_data, test_label = process_tiny_imagenet()
            train_transform = transforms.Compose([transforms.ToTensor(), 
                                        transforms.Normalize(mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262])])
                                        #   transforms.RandomHorizontalFlip(0.5),  mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262]
                                        #   transforms.RandomCrop(size=[32,32], padding=4)])
            test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262])])
            train_indices = random.sample(range(len(train_data)), self.args.datasize)

            train_data = torch.utils.data.TensorDataset(torch.stack(train_data, dim=0).float().view( len(train_data), 3, 64, 64 ), torch.tensor(train_label))
            test_data = torch.utils.data.TensorDataset(torch.stack(test_data, dim=0).float().view(len(test_data), 3, 64, 64 ), torch.tensor(test_label))


        train_loader = DataLoader(torch.utils.data.Subset(train_data, train_indices), batch_size=self.args.batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=self.args.batch_size)
        return train_loader, test_loader
      
    def learn_kernel(self):

        opt = torch.optim.SGD(self.model.parameters(), lr=0.1)
        self.model.train()
        self.args.batch_size = 2048
        train_loader, _ = self.get_loaders()

        for _ in range(self.args.kernel_epoch):
            unif_loss = []
            for batch, (X, y) in enumerate(train_loader):
                
                imgs = X.cuda()
                opt.zero_grad()
                memory = self.model.kernel_forward(imgs)
                loss = 0
                t = 0
                for a in memory:
                    for s in a:
                        t+=1
                        loss += uniform_loss(F.normalize(s, dim=-1))
                loss/=t

                loss.backward()
                unif_loss.append(loss.item())
                opt.step()
            
            print('uniform loss', np.mean(unif_loss))

    def train(self, dataloader):
        
        self.model.train()
        losses = []
        matches = 0
        n_samples = 0

        for batch, (X, y) in enumerate(dataloader):
            imgs, labels = X.cuda(), y.cuda()

            self.optimizer.zero_grad()
            pred = self.model(imgs)
            loss = self.cri(pred, labels)
            matches += (pred.argmax(-1) == labels).sum().item()

            losses.append(loss.item())
            n_samples += X.size(0)

            loss.backward()
            self.optimizer.step()

        return np.mean(losses), matches/n_samples

    def test(self, dataloader):

        self.model.eval()
        losses = []
        matches = 0
        n_samples = 0

        with torch.no_grad():
            for batch, (X,y) in enumerate(dataloader):
                imgs, labels = X.cuda(), y.cuda()

                pred = self.model(imgs)
                loss = self.cri(pred, labels)
                matches += (pred.argmax(-1) == labels).sum().item()
                losses.append(loss.item())
                n_samples += X.size(0)

        return np.mean(losses), matches/n_samples

    def run(self):

        train_loader, test_loader = self.get_loaders()
        self.sch = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.args.epoch, eta_min = 0.0)

        log = {
            'train loss':[],
            'train acc':[],
            'test loss':[],
            'test acc':[],
            'mode':[],
            'epoch':[]
        }

        for e in range(self.args.epoch):

            train_loss, train_acc= self.train(train_loader)
            test_loss, test_acc=self.test(test_loader)

            log['train loss'].append(train_loss)
            log['train acc'].append(train_acc)
            log['test loss'].append(test_loss)
            log['test acc'].append(test_acc)
            log['mode'].append(self.args.mode + f"+ {self.args.activation}")
            log['epoch'].append(e)

            print('[EPOCH]', e, 'Train Loss:', train_loss, 'Train Acc', train_acc*100)
            print('[EPOCH]', e, 'Test Loss:', test_loss, 'Test Acc', test_acc*100)
            print()

            self.sch.step()

        return log

def main():

    plot_data = {
        'train loss':[],
        'train acc':[],
        'test loss':[],
        'test acc':[],
        'mode':[],
        'epoch':[]
    }

    for i in range(3):
        
        args = get_args()
        args.seed = i

        if args.data == 'cifar10' or args.data == 'cifar100':
            args.img_size = 32
        
        elif args.data == 'mnist':
            args.img_size = 28
            args.channel = 1

        elif args.data == 'tiny_imagenet':
            args.img_size = 64
            args.channel = 3
            args.patch_size = 32
        else:
            raise Exception
        
        args.activation = 'softmax1'
        args.mode = 'MHN'
        exp = Exp(args)
        log1 = exp.run()

        args.activation = 'softmax'
        args.mode = 'MHN'
        exp = Exp(args)
        log2 = exp.run()

        args.activation = 'sparsemax'
        args.mode = 'MHN'
        exp = Exp(args)
        log_umhn_sparse = exp.run()

        for k in log1.keys():
            plot_data[k] = plot_data[k] + log2[k]
            plot_data[k] = plot_data[k] + log1[k]
            plot_data[k] = plot_data[k] + log_umhn_sparse[k]
            print(k, '!!', len(plot_data[k]))


    sns.lineplot(data=plot_data, x="epoch", y="train loss", hue="mode")
    plt.savefig(f'imgs_robin/train_loss_{args.data}_{args.datasize}_deep.png')
    plt.clf()

    sns.lineplot(data=plot_data, x="epoch", y="train acc", hue="mode")
    plt.savefig(f'imgs_robin/train_acc_{args.data}_{args.datasize}_deep.png')
    plt.clf()

    sns.lineplot(data=plot_data, x="epoch", y="test loss", hue="mode")
    plt.savefig(f'imgs_robin/test_loss_{args.data}_{args.datasize}_deep.png')
    plt.clf()

    sns.lineplot(data=plot_data, x="epoch", y="test acc", hue="mode")
    plt.savefig(f'imgs_robin/test_acc_{args.data}_{args.datasize}_deep.png')
    plt.clf()

    df = pd.DataFrame(plot_data)
    df.to_csv(f"results_robin/{args.data}_result_warmup_{args.datasize}_deep.csv")

main()