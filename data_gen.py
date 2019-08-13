import sys, os

sys.path.append("../datagen/")
sys.path.append("../models/")
import numpy as np
from torchvision import transforms
import torch
import torch.utils.data as data
from PIL import Image


# datagen class
class Gen(data.Dataset):
    def __init__(self, samples, labels, classes, size=224, multi_task=False, type='train', multi_layer_dict=None):
        super(Gen, self).__init__()
        self.type = type
        self.samples = samples
        # 最初只考虑了单标签（一个样本一个标签的情况）
        # 接下来需要把单标签的情况改成多标签的情况，也就是说labels的每一个元素不再是一个类的名称
        # 而是一个类的list，该list可能只有1个元素，可能包含多个元素
        self.labels = labels
        self.multi_task = multi_task
        self.multi_layer_dict = multi_layer_dict

        # 如果是多层级的，那么使用最后的层级
        if not self.multi_layer_dict is None:
            self.classes = classes[-1]
            self.multi_layer_classes = classes
        else:
            self.classes = classes
            self.multi_layer_classes = None

        if type == 'train':
            self.trans = self.trans_train(size)
        elif type == 'val':
            self.trans = self.trans_val(size)
        elif type == 'norm':
            self.trans = self.trans_norm(size)

        self.len = len(self.samples)

    def trans_train(self, size):
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomRotation(90),
            transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            # 同济patch的参数
            # transforms.Normalize(mean=[0.889, 0.818, 0.869],
            #                      std=[0.025, 0.038, 0.027]),
            # transforms.Normalize(mean=[0.853, 0.779, 0.830],
            #                      std=[0.042, 0.053, 0.045])
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
            # 同济大图的参数
            # transforms.Normalize(mean=[0.937, 0.934, 0.929],
            #                      std=[0.025, 0.027, 0.031])
        ])

    def trans_val(self, size):
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.695, 0.695, 0.694],
            #                      std=[0.119, 0.119, 0.120])
            # transforms.Normalize(mean=[0.889, 0.818, 0.869],
            #                      std=[0.025, 0.038, 0.027]),
            # transforms.Normalize(mean=[0.853, 0.779, 0.830],
            #                      std=[0.042, 0.053, 0.045])
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
            # transforms.Normalize(mean=[0.937, 0.934, 0.929],
            #                      std=[0.025, 0.027, 0.031])
        ])

    def trans_norm(self, size):
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor()
        ])

    def __getitem__(self, idx):
        cur_p = self.samples[idx]
        # 当修改为多标签模式之后，cur_label为一个list，
        # 该list或许包含一个或一个以上的元素，取决于该样本同时属于多个类
        cur_label = self.labels[idx]

        if self.multi_task:
            label = [[0.0, 1.0]] * len(self.classes)
            # 修改成可同时训练为多个类的标签方式
            for cur_l in cur_label:
                cur_ind = self.classes.index(cur_l)
                label[cur_ind] = [1.0, 0.0]
        else:
            label = [0.0] * len(self.classes)
            # 修改成可同时训练为多个类的标签方式
            for cur_l in cur_label:
                if cur_l in self.classes:
                    cur_ind = self.classes.index(cur_l)
                    label[cur_ind] = 1.0

                    # 如果是多层级的
                    if not self.multi_layer_dict is None:
                        labels = []
                        layer_info = self.multi_layer_dict[cur_l]
                        # 遍历每个上层的label，第0个位置为最顶层
                        for s, layer_label in enumerate(layer_info):
                            # 通过得到s层所在的类别的list构建onehot编码
                            tmp_label_onehot = [0.0] * len(self.multi_layer_classes[s])
                            # 同时给出第几个是需要设置为1.0的
                            tmp_ind = self.multi_layer_classes[s].index(layer_label)
                            tmp_label_onehot[tmp_ind] = 1.0
                            # 得到上面各个层的label
                            labels.append(tmp_label_onehot)
                        # 然后加上最底层的label，这时，整个label就构建完成了
                        labels.append(label)

        if not self.multi_layer_classes is None:
            labels = [torch.Tensor(m) for m in labels]
        else:
            labels = torch.Tensor(label)

        img = Image.open(cur_p).convert("RGB")
        img = self.trans(img)
        return img, labels, cur_p

    def collate_fn(self, batch):
        imgs = torch.stack([m[0] for m in batch])
        if not self.multi_layer_dict is None:
            labels = []
            for s in range(len(self.multi_layer_classes)):
                labels.append(torch.stack([m[1][s] for m in batch]))
        else:
            labels = torch.stack([m[1] for m in batch])
        cur_ps = [m[2] for m in batch]
        return imgs, labels, cur_ps

    def __len__(self):
        return len(self.samples)


def data_norm(data_loader):
    t1 = []
    t2 = []
    t3 = []
    for batch_idx, (inputs1, labels, months) in enumerate(data_loader):
        im = inputs1.data.cpu().numpy()
        im = im[0]
        assert im.shape[0] == 3, "做norm计算的图像shape不正确,图像shape为%s" % str(im.shape)
        t1.append(np.mean(im[0, :, :].flatten()) * 1.0)
        t2.append(np.mean(im[1, :, :].flatten()) * 1.0)
        t3.append(np.mean(im[2, :, :].flatten()) * 1.0)

        t1_, s1_ = np.mean(t1), np.std(t1)
        t2_, s2_ = np.mean(t2), np.std(t2)
        t3_, s3_ = np.mean(t3), np.std(t3)

        print('%d, %.3f %.3f %.3f -- %.3f %.3f %.3f' % (batch_idx, t1_, t2_, t3_, s1_, s2_, s3_), end='\r')
    print()


def need_data_norm(data_ps, labels, classes, size):
    # get norm param
    print('Data normalization...')
    norm_gen = Gen(samples=data_ps, labels=labels, classes=classes, size=size, type='norm')
    norm_loader = data.DataLoader(norm_gen, batch_size=1, shuffle=True, num_workers=40,
                                  collate_fn=norm_gen.collate_fn)
    data_norm(norm_loader)


def get_data_loader(ps, size, batch_size, num_workers=40, type='train'):
    gen = Gen(ps, size=size, type=type)
    loader = data.DataLoader(gen, batch_size=int(batch_size), shuffle=True, num_workers=num_workers,
                                 collate_fn=gen.collate_fn)
    num = int(gen.len / batch_size)
    return loader, num
#
#
# def get_data_loader_conf(path_X, y, size, batch_size, num_workers=40):
#     gen = GenConf(path_X, y, size=size, type='train')
#     loader = data.DataLoader(gen, batch_size=batch_size, shuffle=True, num_workers=num_workers,
#                              collate_fn=gen.collate_fn)
#     num = int(gen.len / batch_size)
#     return loader, num
