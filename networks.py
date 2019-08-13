import os, sys
sys.path.append("../datagen/")
sys.path.append("../gyw_utils/")
from model_utils import BranchNet, RootInceptionV3, ClassifierNet
import torch.nn as nn
import torch


class NormalNet(nn.Module):
    def __init__(self, classes, pool_size1, pool_size2, branch_in, multi_task):
        super(NormalNet, self).__init__()

        self.classes = classes
        self.multi_task = multi_task
        self.features = RootInceptionV3(pool_size=pool_size1)
        self.pool = nn.AvgPool2d(kernel_size=pool_size2)
        self.branches = {}
        self.classifiers = {}
        if self.multi_task:
            for name in classes:
                setattr(self, 'classifiers_'+name, ClassifierNet(in_channels=branch_in, num_class=2, pool_size=pool_size2))
        else:
            self.classifier = ClassifierNet(in_channels=branch_in, num_class=len(self.classes), pool_size=pool_size2)

    def forward(self, x):
        x = self.features(x)
        x_out = []
        if self.multi_task:
            for name in self.classes:
                x_out.append(eval('self.classifiers_%s(x)' % name))
        else:
            x_out = self.classifier(x)
        return x_out
#

# 实现一个multi-task，多个分支，每个分支判断一种等级的分类
class MultiLayerNet(nn.Module):
    def __init__(self, multi_layer_classes, pool_size1, pool_size2, branch_in):
        super(MultiLayerNet, self).__init__()
        self.multi_layer_classes = multi_layer_classes
        # 有多少个layer就有多少个分支，每个分支的class的个数对应num_class_each_layer
        self.layer_num = len(multi_layer_classes)
        self.num_class_each_layer = [len(m) for m in multi_layer_classes]
        self.features = RootInceptionV3(pool_size=pool_size1)
        self.pool = nn.AvgPool2d(kernel_size=pool_size2)
        self.branches = {}
        self.classifiers = {}
        for k in range(self.layer_num):
                setattr(self, 'classifiers_'+str(k), ClassifierNet(in_channels=branch_in, num_class=self.num_class_each_layer[k], pool_size=pool_size2))

    def forward(self, x):
        x = self.features(x)
        x_out = []
        for k in range(self.layer_num):
            x_out.append(eval('self.classifiers_%s(x)' % k))
        return x_out


class FixedMultiLayerNet(nn.Module):
    def __init__(self, multi_layer_classes, pool_size1, pool_size2, branch_in):
        super(FixedMultiLayerNet, self).__init__()
        self.multi_layer_classes = multi_layer_classes
        # 有多少个layer就有多少个分支，每个分支的class的个数对应num_class_each_layer
        self.layer_num = len(multi_layer_classes)
        self.num_class_each_layer = [len(m) for m in multi_layer_classes]
        self.features = RootInceptionV3(pool_size=pool_size1)
        self.pool = nn.AvgPool2d(kernel_size=pool_size2)
        self.branches = {}
        self.classifiers = {}
        classifier = [
            nn.Dropout(0.5),
            nn.Linear(in_features=branch_in+self.num_class_each_layer[0], out_features=self.num_class_each_layer[1])]
        self.classifier = nn.Sequential(*classifier)
        self.classifiers_0 = ClassifierNet(in_channels=branch_in, num_class=self.num_class_each_layer[0], pool_size=pool_size2)
        self.classifiers_1 = ClassifierNet(in_channels=branch_in+self.num_class_each_layer[0], num_class=self.num_class_each_layer[1], pool_size=pool_size2)

    def forward(self, x):
        x = self.features(x)
        x_out = []
        x0 = self.classifiers_0(x)
        x1 = self.pool(x)
        x1 = x1.view(x1.size(0), -1)
        x1 = torch.cat([x0, x1], 1)
        x1 = self.classifier(x1)
        x_out.append(x0)
        x_out.append(x1)
        return x_out


