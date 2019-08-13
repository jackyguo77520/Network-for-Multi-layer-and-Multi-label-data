##################### 前提条件
# 1. 数据存放结构，不同类别的数据放入不同的文件夹中
#    data -|
#          | class 1
#          | class 2
#          | ...

import numpy as np
from sklearn.cross_validation import StratifiedKFold
import pickle

class DataUtils:
    def __init__(self, path_dict, classes, n_folds, seed=100, ps_class_dict=None):
        """
        data_info 是一个dict
            {'0fold':   {'class1': [paths], 'class2': [paths], ...
                         class_num: [num_class1, num_class2, ...]，
                         X_val: [paths], y_val: [y]},
             '1fold':   {'class1': [paths], 'class2': [paths], ...
                         class_num: [num_class1, num_class2, ...],
                         X_val: [paths], y_val: [y]},
             ...}
        patch_kfold_dict:
            { '0fold': {'train': {'class1': [patches], 'class2': [patches], ...},
                        'val':   {'class1': [patches], 'class2': [patches], ...}}
              '1fold': {'train': {'class1': [patches], 'class2': [patches], ...},
                        'val':   {'class1': [patches], 'class2': [patches], ...}}
              ...}

        :param path_dict:
                    { 'class1': [p1, p2,...],
                      'class2': [p1, p2,...],
                       ...}
        :param classes:
                    [  class1_name,   class2_name,  ... ]
        :param n_folds:
        :param seed:
        :param ps_class_dict: 这个参数是针对一个样本属于多个类别的情况下存在的
        如果该参数不为None,那么将会给出每一个样本所对应的类别list，形如
                { p1: [c1, c3], p2:[c1], p3:[c5, c7, c9]...}
        """
        self.patch_kfold_dict = {}
        self.classes = classes
        self.seed = seed
        self.data_info = {}
        self.n_folds = n_folds
        self.path_dict = path_dict
        if isinstance(ps_class_dict, str):
            with open(ps_class_dict, 'rb') as f:
                self.ps_class_dict = pickle.load(f)
        else:
            self.ps_class_dict = ps_class_dict

        for cls in classes:
            print(cls, len(path_dict[cls]))

        # 获得所有的paths以及对应的class name
        self.paths_all, self.y_all = self.cat_all_paths()

        # 执行stratifiedKfold
        self.skf = StratifiedKFold(self.y_all, n_folds=n_folds, shuffle=True, random_state=seed)

        # 获取每一个fold的信息，并放入data_info
        self.data_info = self.get_info_for_every_kfold()

    def use_multi_class_label(self, samples, labels):
        """
        此函数是在加入多类别标签之后是必用的函数，其用法应该是在进入到Gen之前的最后一步
        :param samples:
        :param labels:
        :return:
        """
        # 如果存在这个多类别的补充dict,那么样本的label由补充的dict来赋值,其dict对应的value为包含一个或多个类的list
        # 如果不是那么将每一个label元素变成一个list,因为在后续的程序中，处理的label对象为list
        if not self.ps_class_dict is None:
            # 这种label的产生通过sample的名字在self.ps_class_dict字典中查找出来
            new_labels = [self.ps_class_dict[m] for m in samples]
        else:
            # 如果不适用dict，那么就是原来的label变成list
            new_labels = [[m] for m in labels]
        return new_labels

    def set_cur_k(self, kth):
        self.cur_dict = self.data_info['%dfold' % kth]
        self.cur_dict = self.split_data_into_blocks(self.cur_dict)

    def get_info_for_every_kfold(self):
        data_info = {}
        kth = 0
        # 将每一个fold的paths按照类别放入一个新的dict当中
        for train_index, val_index in self.skf:
            print("TRAIN:", len(train_index), "VALIDATION:", len(val_index))
            X_train, X_val = self.paths_all[train_index], self.paths_all[val_index]
            y_train, y_val = self.y_all[train_index], self.y_all[val_index]

            # 每一k折的dict
            cur_kfold_dict = {'class_num': [], 'X_train': X_train, 'y_train': y_train, 'X_val': X_val, 'y_val': y_val}
            # 把每一个path按照不同的类名字放入
            for s, cur_y in enumerate(y_train):
                # 获取当前类名
                cur_class = cur_y
                # 如果当前类名字不在字典中
                if not cur_class in cur_kfold_dict.keys():
                    # 那么将当前的path让入一个list当中
                    cur_kfold_dict[cur_class] = [X_train[s]]
                else:
                    cur_kfold_dict[cur_class].append(X_train[s])

            # 计算每一个类别的样本个数
            for c in self.classes:
                cur_kfold_dict['class_num'].append(len(cur_kfold_dict[c]))

            # 将每个类的数据按照最小类的数据个数一节节的分成多分
            cur_kfold_dict = self.split_data_into_blocks(cur_kfold_dict)

            # 将当前kfold的dict放入到data_info当中
            data_info['%dfold' % kth] = cur_kfold_dict

            kth += 1
        return data_info

    def cat_all_paths(self):
        """
        将所有的paths以及其对应的类别，放到一起，变成
        paths [ p1, p2, ...,p3000, p3001,...,pn]
        y [c1, c1,...,c3, c3,...,cm]
        :return:
        """
        paths = None
        y = None
        flag = False
        for s, c in enumerate(self.classes):
            cur_paths = self.path_dict[c]
            # cur_y = np.ones(len(cur_paths)) * s
            cur_y = np.array([c] * len(cur_paths))
            if not flag:
                paths = cur_paths
                y = cur_y
                flag = True
            else:
                paths = np.concatenate((paths, cur_paths))
                y = np.concatenate((y, cur_y))
        return paths, y

    def split_data_into_blocks(self, cur_dict):
        """
        将每个类的数据按照最小类的数据个数一节节的分成多分。
        :param cur_dict:
        :return:
        """
        min_num = np.min(cur_dict['class_num'])
        blocks_num = []
        for s, c in enumerate(self.classes):
            cur_num = cur_dict['class_num'][s]
            accumulate_num = 0
            blocks = []
            while accumulate_num < cur_num:
                cur_block = cur_dict[c][accumulate_num: min(cur_num, accumulate_num + min_num)]
                accumulate_num += min_num
                if accumulate_num > cur_num:
                    cur_block.extend(cur_dict[c][: accumulate_num - cur_num])
                blocks.append(cur_block)
            if not c + '_blocks' in cur_dict.keys():
                cur_dict[c + '_blocks'] = blocks
            blocks_num.append(len(blocks))
            # print('class %s has %d blocks' % (c, len(blocks)))
        cur_dict['blocks_num'] = blocks_num
        return cur_dict

    def get_balanced_data(self, counter):
        """
        将每一个类别中的样本数目调成一样，该函数为训练时数据的最后出口
        :param counter:
        :return:
        """
        selected_data = []
        selected_class = []
        # cur_kfold_dict = self.data_info['%dfold' % kth]
        for s, c in enumerate(self.classes):
            cur_blocks = self.cur_dict[c + '_blocks']
            cur_blocks_len = len(cur_blocks)
            residule = counter % cur_blocks_len
            selected_block = cur_blocks[residule]
            selected_data.extend(selected_block)
            selected_class.extend([c] * len(selected_block))
        self.selected_data = np.array(selected_data)
        self.selected_class = np.array(selected_class)

    def data_out(self, type):
        # 所有数据
        if type == 'all':
            X = self.paths_all
            y = self.y_all
        # 选择出的数据平衡的训练数据
        elif type == 'selected_train':
            X = self.selected_data
            y = self.selected_class
        # 所有的训练数据（数据有可能不平衡）
        elif type == 'train':
            X = self.cur_dict['X_train']
            y = self.cur_dict['y_train']
        # 验证数据
        elif type == 'val':
            X = self.cur_dict['X_val']
            y = self.cur_dict['y_val']

        # 将这一函数嵌入到该类的数据用作训练的最后出口
        y = self.use_multi_class_label(X, y)
        return X, y

    def patches_aggregation(self, name_dict, max_patch_per_image):
        """
        1. 该函数将分离出来train和val的图片通过name_dict获得他们对应的patches
        2. 然后对于每个图片取最多max_patch_per_image个patches
        3. 之后将train的图片和patch堆叠，以及将val的图片的patch堆叠
        4. 返回train和val以及每个类多少个num的dict，每个fold一个这样的dict
        :param name_dict: {'TCGA-BH-A0EB': [patch1, patch2,...],
                            'TCGA-E2-A15S': [patch1, patch2,...], ...}
        :param max_patch_per_image: 每一个图最多提取多少个patch
        :return:
        """
        patch_kfold_dict = {}
        # 处理每一个kfold
        for kth in range(self.n_folds):
            # 当前fold的dict
            cur_fold_dict = self.data_info['%dfold' % kth]
            train_ims = cur_fold_dict['X_train']
            train_labels = cur_fold_dict['y_train']
            val_ims = cur_fold_dict['X_val']
            val_labels = cur_fold_dict['y_val']

            # # 获取image的名字
            # for s, cls in enumerate(self.classes):
            #     train_names.extend(cur_fold_dict[cls])
            #     # train_labels.extend([s]*len(cur_fold_dict[cls]))
            #     train_labels.extend([cls] * len(cur_fold_dict[cls]))

            patch_dict = {'train': {}, 'val': {}, 'class_num': []}
            for s, cur_im in enumerate(train_ims):
                # 当前图片的类
                cur_cls = train_labels[s]
                # 当前图片的所有patches
                patches = name_dict[cur_im]
                # 打散
                np.random.shuffle(patches)
                # 将打散之后的patch依次堆放入tmp_dic的train中
                if not cur_cls in patch_dict['train'].keys():
                    patch_dict['train'][cur_cls] = []
                # 从每一个image下最多弄出30个patches
                patch_dict['train'][cur_cls].extend(patches[: min(len(patches), max_patch_per_image)])

            # 将val数据进行相同的操作
            for s, cur_im in enumerate(val_ims):
                cur_cls = val_labels[s]
                patches = name_dict[cur_im]
                if not cur_cls in patch_dict['val'].keys():
                    patch_dict['val'][cur_cls] = []
                patch_dict['val'][cur_cls].extend(patches[: min(len(patches), max_patch_per_image)])

            # 统计一下training数据中每个类的sample个数
            for cls in self.classes:
                patch_dict['class_num'].append(len(patch_dict['train'][cls]))

            if not '%dfold' % kth in patch_kfold_dict.keys():
                patch_kfold_dict['%dfold' % kth] = patch_dict

        return patch_kfold_dict


def demo():
    kth = 0
    path_dict = {'small': ['123'] * 25, 'middle': ['456'] * 8, 'large': ['789'] * 15}
    classes = ['small', 'middle', 'large']
    gyw_utils = DataUtils(path_dict=path_dict, classes=classes, n_folds=5, seed=10)
    for counter in range(100000000):
        train_samples, train_labels = gyw_utils.get_balanced_data(kth=kth, counter=counter)
        val_samples = gyw_utils.data_info['%dfold' % kth]['X_val']
        val_labels = gyw_utils.data_info['%dfold' % kth]['y_val']
        # train_gen = Gen(train_samples, train_labels, ...)
        # train_loader = DataLoader(train_gen, ...)
        # val_gen = Gen(val_samples, val_labels, ...)
        # val_loader = DataLoader(val_gen, ...)
        # train(train_loader, ...)
        # val(val_loader, ...)


def demo_patch():
    kth = 0
    path_dict = {'small': ['123'] * 25, 'middle': ['456'] * 8, 'large': ['789'] * 15}
    patch_dict = {'123': ['patch1'] * 10, '456': ['patch2'] * 5, '789': ['patch3'] * 15}

    classes = ['small', 'middle', 'large']
    gyw_utils = DataUtils(path_dict=path_dict, classes=classes, n_folds=5, seed=10)
    patch_kfold_dict = gyw_utils.patches_aggregation(name_dict=patch_dict)


if __name__ == '__main__':
    demo()
