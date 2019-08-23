import torch
from PIL import Image
from torch.utils import data


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)


def default_loader(path):
    return Image.open(path)


class ShoeSourceDataSet(data.Dataset):
    def __init__(self, img_lists, label_lists, img_transform=None, label_transform=None, test=False):
        self.label_lists = label_lists
        self.img_lists = img_lists
        self.test = test
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.files = {'img': [], 'label': []}
        for img_name in img_lists:
            self.files['img'].append(img_name.strip('\n'))
        for label_name in label_lists:
            self.files['label'].append(label_name.strip('\n'))

    def __len__(self):
        return len(self.files['img'])

    def __getitem__(self, item):
        img_path = self.files['img'][item]
        label_path = self.files['label'][item]

        img = Image.open(img_path).convert('RGB')
        label = Image.open(label_path)

        if self.img_transform:
            img = self.img_transform(img)

        if self.label_transform:
            label = self.label_transform(label)

        if self.test:
            return img, torch.Tensor(), img_path
        else:
            return img, label


class ShoeTargetDataSet(data.Dataset):
    def __init__(self, img_lists, img_transform=None, test=False):
        self.img_lists = img_lists
        self.test = test
        self.img_transform = img_transform
        self.files = {'img': [], 'label': []}

        for img_name in img_lists:
            self.files['img'].append(img_name.strip('\n'))

    def __len__(self):
        return len(self.files['img'])

    def __getitem__(self, item):
        img_path = self.files['img'][item]
        img = Image.open(img_path).convert('RGB')

        if self.img_transform:
            img = self.img_transform(img)

        if self.test:
            return img, torch.Tensor(), img_path
        else:
            return img, torch.Tensor()


def get_dataset(dataset_name, img_lists, label_lists, img_transform, label_transform, test=False):
    assert dataset_name in ['source', 'target', 'test']

    if dataset_name == 'source':
        return ShoeSourceDataSet(img_lists=open(img_lists).readlines(),
                                 label_lists=open(label_lists).readlines(),
                                 img_transform=img_transform, label_transform=label_transform,
                                 test=test)
    else:
        # for target set test=False and test set test=True
        return ShoeTargetDataSet(img_lists=open(img_lists).readlines(), img_transform=img_transform, test=test)
