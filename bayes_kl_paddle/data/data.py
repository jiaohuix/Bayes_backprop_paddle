'''
还要改划分数据集
'''
import numpy as np
import paddle
from paddle.io import Dataset
import paddle.vision.transforms as transforms
from paddle.io import Subset  # 用于拆分训练和验证集

class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            sample = self.transform(sample)

        return sample, label


def extract_classes(dataset, classes):
    idx = paddle.zeros_like(dataset.targets, dtype=paddle.bool)
    for target in classes:
        idx = idx | (dataset.targets == target)

    data, targets = dataset.data[idx], dataset.targets[idx]
    return data, targets


def prep_dataset(conf):
    dataset,mean,std =conf['data']['name'],conf['data']['mean'],conf['data']['std']
    size=conf['data']['input_size']
    mnist_size=(size[1],size[2])
    transform_split_mnist = transforms.Compose([
        transforms.Resize(mnist_size),
        transforms.Normalize(mean=mean, std=std, data_format='CHW')
    ])

    transform_mnist = transforms.Compose([
        transforms.Resize(mnist_size),
        transforms.Normalize(mean=mean, std=std, data_format='CHW')

    ])

    transform_cifar = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean=mean, std=std)
    ])
    if (dataset == 'CIFAR10'):
        trainset = paddle.vision.datasets.Cifar10(download=True, transform=transform_cifar)
        testset = paddle.vision.datasets.Cifar10(download=True, transform=transform_cifar)
        num_classes = 10
        in_channels = 3

    elif (dataset == 'CIFAR100'):
        trainset = paddle.vision.datasets.Cifar100(data_file='./data', mode='train', download=True,
                                                   transform=transform_cifar)
        testset = paddle.vision.datasets.Cifar100(data_file='./data', mode='test', download=True,
                                                  transform=transform_cifar)
        num_classes = 100
        in_channels = 3

    elif (dataset == 'MNIST'):
        trainset = paddle.vision.datasets.MNIST(mode='train', download=True, transform=transform_mnist)
        testset = paddle.vision.datasets.MNIST(mode='test', download=True, transform=transform_mnist)
        num_classes = 10
        in_channels = 1

    elif (dataset == 'SplitMNIST-2.1'):
        trainset = paddle.vision.datasets.MNIST(mode='train', download=True, transform=transform_mnist)
        testset = paddle.vision.datasets.MNIST(mode='test', download=True, transform=transform_mnist)

        train_data, train_targets = extract_classes(trainset, [0, 1, 2, 3, 4])
        test_data, test_targets = extract_classes(testset, [0, 1, 2, 3, 4])

        trainset = CustomDataset(train_data, train_targets, transform=transform_split_mnist)
        testset = CustomDataset(test_data, test_targets, transform=transform_split_mnist)
        num_classes = 5
        in_channels = 1

    elif (dataset == 'SplitMNIST-2.2'):
        trainset = paddle.vision.datasets.MNIST(mode='train', download=True, transform=transform_mnist)
        testset = paddle.vision.datasets.MNIST(mode='test', download=True, transform=transform_mnist)

        train_data, train_targets = extract_classes(trainset, [5, 6, 7, 8, 9])
        test_data, test_targets = extract_classes(testset, [5, 6, 7, 8, 9])
        train_targets -= 5  # Mapping target 5-9 to 0-4
        test_targets -= 5  # Hence, add 5 after prediction

        trainset = CustomDataset(train_data, train_targets, transform=transform_split_mnist)
        testset = CustomDataset(test_data, test_targets, transform=transform_split_mnist)
        num_classes = 5
        in_channels = 1

    elif (dataset == 'SplitMNIST-5.1'):
        trainset = paddle.vision.datasets.MNIST(mode='train', download=True, transform=transform_mnist)
        testset = paddle.vision.datasets.MNIST(mode='test', download=True, transform=transform_mnist)

        train_data, train_targets = extract_classes(trainset, [0, 1])
        test_data, test_targets = extract_classes(testset, [0, 1])

        trainset = CustomDataset(train_data, train_targets, transform=transform_split_mnist)
        testset = CustomDataset(test_data, test_targets, transform=transform_split_mnist)
        num_classes = 2
        in_channels = 1

    elif (dataset == 'SplitMNIST-5.2'):
        trainset = paddle.vision.datasets.MNIST(mode='train', download=True, transform=transform_mnist)
        testset = paddle.vision.datasets.MNIST(mode='test', download=True, transform=transform_mnist)

        train_data, train_targets = extract_classes(trainset, [2, 3])
        test_data, test_targets = extract_classes(testset, [2, 3])
        train_targets -= 2  # Mapping target 2-3 to 0-1
        test_targets -= 2  # Hence, add 2 after prediction

        trainset = CustomDataset(train_data, train_targets, transform=transform_split_mnist)
        testset = CustomDataset(test_data, test_targets, transform=transform_split_mnist)
        num_classes = 2
        in_channels = 1

    elif (dataset == 'SplitMNIST-5.3'):
        trainset = paddle.vision.datasets.MNIST(mode='train', download=True, transform=transform_mnist)
        testset = paddle.vision.datasets.MNIST(mode='test', download=True, transform=transform_mnist)

        train_data, train_targets = extract_classes(trainset, [4, 5])
        test_data, test_targets = extract_classes(testset, [4, 5])
        train_targets -= 4  # Mapping target 4-5 to 0-1
        test_targets -= 4  # Hence, add 4 after prediction

        trainset = CustomDataset(train_data, train_targets, transform=transform_split_mnist)
        testset = CustomDataset(test_data, test_targets, transform=transform_split_mnist)
        num_classes = 2
        in_channels = 1

    elif (dataset == 'SplitMNIST-5.4'):
        trainset = paddle.vision.datasets.MNIST(mode='train', download=True, transform=transform_mnist)
        testset = paddle.vision.datasets.MNIST(mode='test', download=True, transform=transform_mnist)

        train_data, train_targets = extract_classes(trainset, [6, 7])
        test_data, test_targets = extract_classes(testset, [6, 7])
        train_targets -= 6  # Mapping target 6-7 to 0-1
        test_targets -= 6  # Hence, add 6 after prediction

        trainset = CustomDataset(train_data, train_targets, transform=transform_split_mnist)
        testset = CustomDataset(test_data, test_targets, transform=transform_split_mnist)
        num_classes = 2
        in_channels = 1

    elif (dataset == 'SplitMNIST-5.5'):
        trainset = paddle.vision.datasets.MNIST(mode='train', download=True, transform=transform_mnist)
        testset = paddle.vision.datasets.MNIST(label_path='./data', mode='test', download=True,
                                               transform=transform_mnist)

        train_data, train_targets = extract_classes(trainset, [8, 9])
        test_data, test_targets = extract_classes(testset, [8, 9])
        train_targets -= 8  # Mapping target 8-9 to 0-1
        test_targets -= 8  # Hence, add 8 after prediction

        trainset = CustomDataset(train_data, train_targets, transform=transform_split_mnist)
        testset = CustomDataset(test_data, test_targets, transform=transform_split_mnist)
        num_classes = 2
        in_channels = 1

    return trainset, testset, in_channels, num_classes


def prep_loader(conf,trainset, testset):
    num_workers=conf['data']['num_workers']
    valid_size=conf['data']['valid_size']
    batch_size=conf['hparas']['batch_size']
    # 划分训练和验证集
    num_train = len(trainset)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train)) if valid_size<1 and valid_size>0  else valid_size
    train_idx, valid_idx = indices[split:], indices[:split]  # 前split个作val

    train_subset = Subset(dataset=trainset, indices=train_idx)  # subset是dataset不是sampler
    valid_subset = Subset(dataset=trainset, indices=valid_idx)

    # 创建dataloader
    train_loader = paddle.io.DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                                        num_workers=num_workers)
    valid_loader = paddle.io.DataLoader(valid_subset, batch_size=batch_size,
                                        num_workers=num_workers)
    test_loader = paddle.io.DataLoader(testset, batch_size=batch_size,
                                       num_workers=num_workers)

    return train_loader, valid_loader, test_loader
