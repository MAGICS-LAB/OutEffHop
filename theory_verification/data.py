import torch
import numpy as np
import torch.utils.data
from torchvision import datasets, transforms
import torch.nn.functional as F
import random
from datasets import load_dataset
#from data_utils import Textset, create_vocab, data_preprocessing
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt 
import torch
import torchvision 
from torchvision import transforms
import time

def load_mnist(batch_size, return_loader=False):
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.MNIST(root='./mnist_data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True)
    trainset = list(iter(trainloader))

    testset = datasets.MNIST(root='./mnist_data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=True)
    testset = list(iter(testloader))
    if not return_loader:
        return trainset, testset
    else:
        return trainloader, testloader

def load_cifar10(batch_size, return_loader=False):

    transform = transforms.Compose([transforms.ToTensor()])#, transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./cifar_data', train=True,
                                            download=True,transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                      shuffle=True)
    train_data = list(iter(trainloader))
    testset = torchvision.datasets.CIFAR10(root='./cifar_data', train=False,
                                            download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                      shuffle=True)
    test_data = list(iter(testloader))

    if not return_loader:
        return train_data, test_data
    else:
        return trainloader, testloader

def get_id_dictionary(path):
    id_dict = {}
    for i, line in enumerate(open( path + 'wnids.txt', 'r')):
        id_dict[line.replace('\n', '')] = i
    return id_dict
  
def get_class_to_id_dict(path):
    id_dict = get_id_dictionary()
    all_classes = {}
    result = {}
    for i, line in enumerate(open( path + 'words.txt', 'r')):
        n_id, word = line.split('\t')[:2]
        all_classes[n_id] = word
    for key, value in id_dict.items():
        result[value] = (key, all_classes[key])      
    return result

def get_data(path,id_dict):
    print('starting loading data')
    train_data, test_data = [], []
    train_labels, test_labels = [], []
    t = time.time()
    for key, value in id_dict.items():
        train_data += [plt.imread( path + 'train/{}/{}_{}.JPEG'.format(key, key, str(i)), format='RGB') for i in range(500)]
        train_labels_ = np.array([[0]*200]*500)
        train_labels_[:, value] = 1
        train_labels += train_labels_.tolist()

    for line in open( path + 'val/val_annotations.txt'):
        img_name, class_id = line.split('\t')[:2]
        test_data.append(plt.imread( path + 'val/{}/{}'.format(class_id, img_name) ,format='RGB'))
        test_labels_ = np.array([[0]*200])
        test_labels_[0, id_dict[class_id]] = 1
        test_labels += test_labels_.tolist()

    print('finished loading data, in {} seconds'.format(time.time() - t))
    return train_data, train_labels, test_data, test_labels

def parse_train_data(train_images, train_labels,N_imgs):
    images = torch.zeros((N_imgs, 64,64,3))
    for i,(img,label) in enumerate(zip(train_images, train_labels)):
        if i >= N_imgs:
            break
        if len(img.shape) == 3:
            images[i,:,:,:] = torch.tensor(img,dtype=torch.float) / 255.0 # normalize
    return torch.tensor(images)

def process_tiny_imagenet():

    path = './tiny-imagenet-200/'
    id_dict = get_id_dictionary(path)
    print('starting loading data')
    train_data, test_data = [], []
    train_labels, test_labels = [], []
    t = time.time()
    for key, value in id_dict.items():
        for i in range(500):
            img = plt.imread( path + 'train/{}/{}_{}.JPEG'.format(key, key, str(i)), format='RGB')
            if len(img.shape) == 3:
                train_data.append(torch.tensor(img))
                train_labels.append(int(value))         

    for line in open( path + 'val/val_annotations.txt'):
        img_name, class_id = line.split('\t')[:2]
        img = plt.imread( path + 'val/{}/{}'.format(class_id, img_name) ,format='RGB')
        if len(img.shape) == 3:
            test_data.append( torch.tensor(img))
            test_labels.append(id_dict[class_id])

    return train_data, train_labels, test_data, test_labels


def load_tiny_imagenet(N_imgs, return_loader=False):
    path = './tiny-imagenet-200/'
    train_data, train_labels, test_data, test_labels = get_data(path,get_id_dictionary(path))
    train_images = parse_train_data(train_data, train_labels,len(train_data))
    test_images = parse_train_data(test_data, test_labels,len(test_labels))

    trainset = torch.utils.data.TensorDataset(train_images, torch.tensor(train_labels))
    testset = torch.utils.data.TensorDataset(test_images, torch.tensor(test_labels))

    if return_loader:
        return train_data, train_labels, test_data, test_labels

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=N_imgs,
                                      shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=N_imgs,
                                      shuffle=False)
    testset = list(iter(testloader))
    trainset = list(iter(trainloader))

    if not return_loader:
        return trainset, testset
    else:
        return trainloader, testloader
    
def load_synthetic(N_imgs):

    image_list = []
    for i in range(N_imgs):
        # indices = np.random.choice(100, signal, replace=False)  # Generate 10 random indices from 0 to 99
        # rows, cols = np.unravel_index(indices, (10, 10))  # Convert indices to 2D coordinates
        # img_array = np.random.normal(0, 0., (image_width, image_length))

        # for i in range(signal):
        #     img_array[rows[i], cols[i]] = np.random.choice([-1, 1])

        img = np.random.normal(size=100)
        image_list.append(torch.from_numpy(img))
    
    return torch.stack(image_list, dim=0)

def get_text_data(args):

    tf = 'text'
    train_data = load_dataset(args.data, split='train')
    test_data = load_dataset(args.data, split='test')

    train_text = [b[tf] for b in train_data]
    test_text = [b[tf] for b in test_data]
    train_label = [b['label'] for b in train_data]
    test_label = [b['label'] for b in test_data]
    clean_train = [data_preprocessing(t, True) for t in train_text]
    clean_test = [data_preprocessing(t, True) for t in test_text]

    vocab = create_vocab(clean_train)
    trainset = Textset(clean_train, train_label, vocab, args.max_len)
    testset = Textset(clean_test, test_label, vocab, args.max_len)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, collate_fn = trainset.collate, shuffle=True)
    test_loader = DataLoader(testset, batch_size=args.batch_size, collate_fn = testset.collate)

    return train_loader, test_loader, trainset, testset