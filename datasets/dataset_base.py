"""
-----------------------------------------------------------------------------------
--  @file       MLfog_lib->dataset_base.py
--  @author     Ma Haoming(En:louis)(https://github.com/evavoid)
--  @brief      xxxxxxxxx
    
--  @Ide        PyCharm
--  @time       2020/12/11-23:41
-----------------------------------------------------------------------------------
"""
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


def DatasetGet(args):
    dataset = {}

    transmnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    dataset["mnist_train_origin"] = datasets.MNIST('./datasets/mnist/', train=True, download=True, transform=transmnist)
    dataset["fmnist_train_origin"] = datasets.FashionMNIST('./datasets/fmnist/', train=True, download=True,
                                                           transform=transmnist)
    dataset["mnist_test_origin"] = datasets.MNIST('./datasets/mnist/', train=False, download=True, transform=transmnist)
    dataset["fmnist_test_origin"] = datasets.FashionMNIST('./datasets/fmnist/', train=False, download=True,
                                                          transform=transmnist)

    dataset["mnist_train_users"] = random_split(dataset['mnist_train_origin'],
                                                [int(len(dataset["mnist_train_origin"]) / args.musers_num) for i in
                                                 range(args.musers_num)])
    dataset["fmnist_train_users"] = random_split(dataset["fmnist_train_origin"],
                                                 [int(len(dataset["fmnist_train_origin"]) / args.fusers_num) for i in
                                                  range(args.fusers_num)])


    return dataset
