from config import *
import os
import matplotlib
matplotlib.use('Agg')
from glob import glob
import torch.utils.data as data
from kfold_utils import DataUtils
from data_gen import Gen
from model_utils import NetOpe, get_save_model_path
from data_cleaner import get_image_label_over_all_folder
from networks import FixedMultiLayerNet


def main():
    # prepare multi label
    # ps_class_dict = get_image_label_over_all_folder(root_folder=train_root, save_path='ps_class.pkl', load=True)

    # prepare DataUtils data format
    class_dict = {}
    for cls in classes_last:
        class_dict[cls] = sorted(glob(os.path.join(train_root + cls, '*')))

    # 使用 multi label
    data_utils = DataUtils(path_dict=class_dict, classes=classes_last, n_folds=5, seed=10, ps_class_dict='ps_class.pkl')

    # 遍历k个fold
    for kth in range(n_folds):
        print('The %dth fold ...' % kth)

        # 获得当前模型的名称
        model_path = get_save_model_path(name=model_name, day=day, k=kth, loop=0)

        print('model save path is: ', model_path)

        # 初始化pretrained inceptionv3网络
        # net = EyeNet(classes=classes, pool_size1=1, pool_size2=8, branch_in=2048, multi_task=multi_task)
        # net = MultiLayerNet(multi_layer_classes=classes_all, pool_size1=1, pool_size2=8, branch_in=2048)
        net = FixedMultiLayerNet(multi_layer_classes=classes_all, pool_size1=1, pool_size2=8, branch_in=2048)
        netope = NetOpe(net, num_class=len(classes_last), classes=classes_all, size=size, model_path=model_path,
                        multi_task=multi_task, resume=False, always_save=False, multi_layer=multi_layer)

        # 设定当前的k折
        data_utils.set_cur_k(kth)
        # 获得validation数据
        val_samples, val_labels = data_utils.data_out(type='val')
        # generator和 data loader
        val_gen = Gen(val_samples, val_labels, classes=classes_all, multi_task=multi_task, size=size,
                      type='val', multi_layer_dict=multi_layer_dict)
        val_loader = data.DataLoader(val_gen, batch_size=int(batch_size), shuffle=True, num_workers=num_workers,
                                     collate_fn=val_gen.collate_fn)
        val_num = int(val_gen.len / batch_size)

        # training
        for epoch in range(epochs):
            print('train')
            for counter in range(times_per_epoch):
                # 计算当前的counter
                counter = epoch * times_per_epoch + counter

                # 为了每个子类的数据在训练的过程中保持平衡的状态，因此把每个类的数据都切成了若干个block
                # 所以counter是用来控制block轮循的，以此来控制出来的是哪一组各类的block
                data_utils.get_balanced_data(counter)

                # 设定好之后，取出当前的training数据
                train_samples, train_labels = data_utils.data_out(type='selected_train')

                # training的generator和data loader
                train_gen = Gen(train_samples, train_labels, classes=classes_all, multi_task=multi_task, size=size,
                                type='train', multi_layer_dict=multi_layer_dict)

                # data loader
                train_loader = data.DataLoader(train_gen, batch_size=int(batch_size), shuffle=True,
                                               num_workers=num_workers,
                                               collate_fn=train_gen.collate_fn)

                # 获得一个epoch需要训练的次数
                train_num = int(train_gen.len / batch_size)

                # 训练
                netope.train(epoch=counter, train_loader=train_loader, num=train_num,
                             epoch_th=epoch_th * times_per_epoch)

            # validation,按照最佳auc的方式选择模型
            netope.val(epoch=counter, test_loader=val_loader, num=val_num, metrics='auc')


if __name__ == '__main__':
    main()
