import os, sys
import math

sys.path.append("../datagen/")
sys.path.append("../models/")
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models
from torchvision import models as torchvision_models
from cnn_finetune import make_model
from torchvision.models.resnet import Bottleneck
from sklearn.metrics import auc
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
# import torch.utils.model_zoo as model_zoo
# from shutil import copy, move
from tqdm import tqdm
from sklearn.metrics.ranking import _binary_clf_curve
from sklearn.metrics import roc_curve, roc_auc_score
import shutil
from grad_cam import GradCam, GuidedBackpropReLUModel, show_cam_on_image
import cv2

colors = ['black', 'darkorange', 'forestgreen', 'cornflowerblue', 'lightcoral', 'darkorchid',
          'brown', 'darkred', 'darkcyan', 'violet', 'lightsalmon', 'deeppink', 'deepskyblue',
          'aqua', 'gold', 'fuchsia', 'lightseagreen']


def spec_sens(y_true, y_pred, pos_label=None, sample_weight=None):
    # get false positive and true positive
    fps, tps, thresholds = _binary_clf_curve(y_true, y_pred,
                                             pos_label=pos_label,
                                             sample_weight=sample_weight)
    # positive sample number
    actual_p = sum(y_true)
    # negative sample number
    actual_f = len(y_true) - sum(y_true)
    fps = fps * 1.0 / actual_f
    tps = tps * 1.0 / actual_p
    # true negative
    tns = 1 - fps
    # specificity and sensitivity
    spec = tns / (fps + tns)
    sens = tps / tps[-1]

    dis = []
    for k in range(len(sens)):
        d = (1 - spec[k]) * (1 - spec[k]) + (1 - sens[k]) * (1 - sens[k])
        dis.append(d)
    index = np.argmin(dis)

    return spec, sens, thresholds, index


def get_save_model_path(name='patch', day='20180904', k=1, loop=0):
    return 'checkpoint/' + name + '_' + day + '_k%d_loop%d.pth' % (k, loop)
    # return './checkpoint/' + name + '_' + day + '_k%d_epoch3.pth' % k


class NetOpe:
    def __init__(self, net, num_class, classes, size, model_path, multi_task=False, resume=False, always_save=False, multi_layer=False):
        self.always_save = always_save
        self.size = size
        self.multi_layer = multi_layer

        if self.multi_layer:
            self.classes = classes[-1]
            self.multi_layer_classes = classes
        else:
            self.classes = classes

        self.model_path = model_path
        self.resume = resume
        self.start_epoch = 0
        self.multi_task = multi_task
        self.best_loss = 1000000
        # self.net = HEPatchNet(classifier_input=classifier_input, num_class=num_class, size=size)
        self.net = net
        # self.net = Inception3(num_classes=num_class, aux_logis=False)
        # self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))
        if not net is None:
            self.net.cuda()

        if self.multi_layer:
            self.criterion = []
            for k, cla in enumerate(self.multi_layer_classes):
                self.criterion.append(FocalLoss(class_num=len(cla), alpha=None, gamma=2, size_average=True))
        else:
            # self.criterion = FocalLoss(class_num=num_class, alpha=torch.Tensor([4, 1]), gamma=2, size_average=True)
            self.criterion = FocalLoss(class_num=num_class, alpha=None, gamma=2, size_average=True)
            # self.criterion = nn.CrossEntropyLoss(size_average=True)
        if self.resume and self.net is not None:
            self.start_epoch, self.best_loss = self.resume_model(self.model_path)


    def _update_model_param(self, state_dict, prefix):
        # 新模型参数dict
        net_state_dict = self.net.state_dict()
        # 将不用的key剥离
        state_dict = {prefix + k: v for k, v in state_dict.items() if prefix + k in net_state_dict}
        # print('same param names: ', state_dict.keys())
        # 更新到新模型中
        net_state_dict.update(state_dict)
        # 新模型加载参数
        self.net.load_state_dict(net_state_dict)

    def resume_model(self, model_path):
        print('==> Resuming from checkpoint')
        checkpoint = torch.load(model_path)
        self.net.load_state_dict(checkpoint['net'])
        # torch.save(self.net, 'model_retinal.pkl')
        # print('model saved')
        # exit()
        best_loss = checkpoint['loss']
        start_epoch = checkpoint['epoch']
        print('start_epoch:', start_epoch)
        return start_epoch, best_loss

    # 动态调整学习率
    def adjust_optim(self, epoch, epoch_th=[5, 15]):
        # lr = 1e-4 if epoch < epoch_th else 1e-5
        if epoch < epoch_th[0]:
            lr = 1e-4
        elif epoch < epoch_th[1]:
            lr = 1e-5
        else:
            lr = 1e-6

        if epoch == epoch_th[0]:
            print('learning rate: 1e-5')
            self.start_epoch, self.best_loss = self.resume_model(self.model_path)
        elif epoch == epoch_th[1]:
            print('learning rate: 1e-6')
            self.start_epoch, self.best_loss = self.resume_model(self.model_path)
        # chosen_param, unchosen_param = self.split_param([self.net.classifier1.parameters()])
        # optimizer_pt = torch.optim.RMSprop([{'params': unchosen_param, 'lr': lr / 10},
        #                                     {'params': chosen_param, 'lr': lr}],
        #                                    lr=lr, momentum=0.9, weight_decay=0.0005)
        # optimizer_pt = torch.optim.RMSprop(filter(lambda p: p.requires_grad, self.net.parameters()),
        #                                    lr=lr, momentum=0.9, weight_decay=0.0005)
        optimizer_pt = torch.optim.Adam(filter(lambda p: p.requires_grad, self.net.parameters()),
                                        lr=lr, weight_decay=0.0005)
        return optimizer_pt

    # 选择fine tune的参数
    def split_param(self, chosen_param):
        chosen_id = []
        for cp in chosen_param:
            chosen_id += list(map(id, cp))
        chosen_p = filter(lambda p: id(p) in chosen_id, self.net.parameters())
        unchosen_p = filter(lambda p: id(p) not in chosen_id, self.net.parameters())
        return chosen_p, unchosen_p

    # training function
    def train(self, epoch, train_loader, num, epoch_th):
        # print('train')
        self.net.train()

        for param in self.net.parameters():
            param.requires_grad = True
        # 只能生效一次,在进行下面param.requires_grad = False操作后，unchosen_param 变为0,暂时还不知道是为什么
        # 但是如果再一次想操作unchosen_param就要重新在执行下面一行代码,并且如果下面的代码在self.net.train()之前
        # 那么经过self.net.train()之后，chosen_param和unchosen_param变为0,一定要注意这一点。
        # chosen_param, unchosen_param = self.split_param([self.net.features.parameters()])

        if epoch < epoch_th[0]:
            for param in self.net.features.parameters():
                param.requires_grad = False
        if epoch == epoch_th[0]:
            print('unfreeze all parameters')

        optimizer_ft = self.adjust_optim(epoch, epoch_th)
        _ = self._train_val(epoch, train_loader, num, optimizer_ft=optimizer_ft, type='train')

    # validation function
    def val(self, epoch, test_loader, num, metrics='loss'):
        print('val')
        # 当只做inference的时候，一下这段最好一起出现
        self.net.eval()
        for param in self.net.parameters():
            param.requires_grad = False

        # 选择metrics为loss或者为auc
        if metrics == 'loss':
            ave_loss = self._train_val(epoch, test_loader, num, optimizer_ft=None, type='val')
            ave_loss = np.mean(ave_loss)
        elif metrics == 'auc':
            predict_dict, predict_label_dict, predict_middle_vector = \
                self._train_val(epoch, test_loader, num, optimizer_ft=None, type='predict')

            if self.multi_layer:
                str_sens, str_spes, str_thre, str_aucc = '', '', '', ''
                for layer in range(len(self.multi_layer_classes)):
                    best_sens, best_spec, best_thre, aucc = self.sens_spec_for_dict(predict_dict, predict_label_dict, layer)
                    ave_loss = 1 - np.mean(aucc)
                    str_sens = str_sens + 'l%d ' % layer + ','.join(['%.3f' % m for m in best_sens]) + '-'
                    str_spes = str_spes + 'l%d ' % layer + ','.join(['%.3f' % m for m in best_spec]) + '-'
                    str_thre = str_thre + 'l%d ' % layer + ','.join(['%.3f' % m for m in best_thre]) + '-'
                    str_aucc = str_aucc + 'l%d ' % layer + ','.join(['%.3f' % m for m in aucc]) + '-'

                print('epoch%d |ave_auc: %.3f' % (epoch, 1-ave_loss),
                      '|sens:', str_sens, '|spes:', str_spes, '|thre:', str_thre, '|aucc:', str_aucc, end='\r')



            else:
                best_sens, best_spec, best_thre, aucc = self.sens_spec_for_dict(predict_dict, predict_label_dict)
                ave_loss = 1 - np.mean(aucc)

                str_sens = ','.join(['%.3f' % m for m in best_sens])
                str_spes = ','.join(['%.3f' % m for m in best_spec])
                str_thre = ','.join(['%.3f' % m for m in best_thre])
                str_aucc = ','.join(['%.3f' % m for m in aucc])
                print('epoch%d |ave_auc: %.3f' % (epoch, 1-ave_loss),
                      '|sens:', str_sens, '|spes:', str_spes, '|thre:', str_thre, '|aucc:', str_aucc, end='\r')

        # 是否更新模型
        update = False
        if ave_loss < self.best_loss:
            update = True
            # if True:
            print('saving...')
            state = {
                'net': self.net.state_dict(),
                'loss': ave_loss,
                'epoch': epoch
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            if self.always_save:
                torch.save(state, self.model_path.split('.pth')[0] + '_epoch%d.pth' % epoch)
            else:
                torch.save(state, self.model_path)
            self.best_loss = ave_loss

        else:
            print()
        return update

    def predict(self, test_loader, num):
        print('predict')
        self.net.eval()
        for param in self.net.parameters():
            param.requires_grad = False
        predict_dict, predict_label_dict, predict_middle_vector = self._train_val(epoch=0, data_loader=test_loader,
                                                                                  num=num, optimizer_ft=None,
                                                                                  type='predict')
        return predict_dict, predict_label_dict, predict_middle_vector

    def sens_spec_for_dict_list(self, predict_dict, predict_labels_dict, classes):
        pred = []
        true = []
        for name in predict_dict.keys():
            # 当前path name的pred，以及path name的true
            p = predict_dict[name]
            t = predict_labels_dict[name]
            pred.append(p)
            true.append(t)
        sens, spec, thre, aucc = {}, {}, {}, {}
        for s, cls in enumerate(classes):
            sp, se, th, best_ind = spec_sens(np.array(true)[:, s], np.array(pred)[:, s], pos_label=None)
            au = auc(sp, se)
            # plt.plot(sp, se, color='g', label=r'AUC = %0.4f' % (au),
            #          lw=1.5, alpha=0.8)
            sens[cls] = se
            spec[cls] = sp
            thre[cls] = th[best_ind]
            aucc[cls] = au
        return sens, spec, thre, aucc

    def confusion_matrix_for_dict_list(self, predict_dict, predict_labels_dict, thre, classes):
        pred = []
        true = []
        for name in predict_dict.keys():
            pred.append(predict_dict[name])
            true.append(predict_labels_dict[name])

        binary_cnf_matrix = {}
        for s, cls in enumerate(classes):
            y_t = np.array(true)[:, s]
            tmp_y_p = np.array(pred)[:, s]
            th = thre[cls]
            y_p = np.zeros(tmp_y_p.shape, dtype=np.int)
            y_p[tmp_y_p > th] = int(1)
            binary_cnf_matrix[cls] = confusion_matrix(y_t, y_p)

        # 最大的两个都是有效的
        ths = [thre[cls] for cls in classes]
        pred = np.ceil([m - ths for m in pred])
        for s in range(len(pred)):
            cur_p = pred[s]
            cur_t = true[s]
            re = np.sum([cur_p[m] * cur_t[m] for m in range(len(cur_p))])
            if re > 0:
                pred[s] = cur_t

        y_pred = np.argmax(pred, axis=1)
        y_true = np.argmax(true, axis=1)
        # y_pred = np.array(pred)
        # for s, cls in enumerate(self.classes):
        #     tmp_y_p = np.array(pred)[:, s]
        #     th = thre[cls]
        #     y_p = np.zeros(tmp_y_p.shape, dtype=np.int)
        #     y_p[tmp_y_p > th] = int(1)
        #     y_pred[:, s] = y_p
        # y_true = np.array(true)
        multi_class_cnf_matrix = confusion_matrix(y_true, y_pred)

        return binary_cnf_matrix, multi_class_cnf_matrix

    def tsne(self, predict_dict, predict_labels_dict, kth, color_list, classes=None, save_dir=None, save_name=None):
        pred = []
        true = []
        for name in predict_dict.keys():
            pred.append(predict_dict[name])
            true.append(predict_labels_dict[name])

        ts = TSNE(n_components=2, perplexity=15, init='random', random_state=None,
                  learning_rate=100, n_iter=1000, n_iter_without_progress=100, min_grad_norm=1e-5, method='exact')
        pred = ts.fit_transform(np.array(pred))
        true = np.array([np.argmax(m) for m in true])
        tmp_ = {}
        for s, t in enumerate(true):
            class_name = classes[t]
            if not class_name in tmp_.keys():
                tmp_[class_name] = [pred[s]]
            else:
                tmp_[class_name].append(pred[s])

        # for s, cls in enumerate(tmp_.keys()):
        for s, cls in enumerate(classes):
            p1 = np.array(tmp_[cls])[:, 0]
            p2 = np.array(tmp_[cls])[:, 1]
            plt.scatter(p1, p2, color=color_list[s], marker='.', linewidths=0.01, alpha=1)
            print(cls, color_list[s])

        if not save_dir is None:
            plt.savefig('figs/%s/%s.jpg' % (save_dir, save_name))
        else:
            plt.savefig('figs/tsen_%dfold.png' % kth)
        plt.close()

    def confidence_interval(self, predict_dict, predict_labels_dict, classes):
        pred = []
        true = []
        for name in predict_dict.keys():
            pred.append(predict_dict[name])
            true.append(predict_labels_dict[name])

        confidence = {}
        for s, cls in enumerate(classes):
            y_t = np.array(true)[:, s]
            y_p = np.array(pred)[:, s]
            # print("Original ROC area: {:0.3f}".format(roc_auc_score(y_t, y_p)))
            n_bootstraps = 1000
            rng_seed = 42  # control reproducibility
            bootstrapped_scores = []
            rng = np.random.RandomState(rng_seed)
            for i in range(n_bootstraps):
                # bootstrap by sampling with replacement on the prediction indices
                indices = rng.random_integers(0, len(y_p) - 1, len(y_p))
                if len(np.unique(y_t[indices])) < 2:
                    # We need at least one positive and one negative sample for ROC AUC
                    # to be defined: reject the sample
                    continue
                score = roc_auc_score(y_t[indices], y_p[indices])
                bootstrapped_scores.append(score)
                # print("Bootstrap #{} ROC area: {:0.3f}".format(i + 1, score))

            # plt.hist(bootstrapped_scores, bins=50)
            # plt.title('Histogram of the bootstrapped ROC AUC scores')
            # plt.show()
            sorted_scores = np.array(bootstrapped_scores)
            sorted_scores.sort()

            # Computing the lower and upper bound of the 90% confidence interval
            # You can change the bounds percentiles to 0.025 and 0.975 to get
            # a 95% confidence interval instead.
            confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
            confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]
            # print("Confidence interval for the score: [{:0.3f} - {:0.3}]".format(
            #     confidence_lower, confidence_upper))
            confidence[cls] = [confidence_lower, confidence_upper]
        return confidence

    def sens_spec_for_dict(self, predict_dict, predict_labels_dict, layer=-1):
        """
        这个函数输入为以path为key，predict的prob为value的dict
        以及label为value的dict
        输出为sens的list，spec的list，thre的list以及auc的list
        list的长度为类别的长度
        """
        pred = []
        true = []
        for name in predict_dict.keys():
            if layer > -1:
                p = predict_dict[name][layer]
                t = predict_labels_dict[name][layer]
            else:
                p = predict_dict[name]
                t = predict_labels_dict[name]
            pred.append(p)
            true.append(t)
        # print(pred[:10], true[:10])
        sens, spec, thre, aucc = [], [], [], []
        if self.multi_layer:
            for s, cls in enumerate(self.multi_layer_classes[layer]):
                sp, se, th, best_ind = spec_sens(np.array(true)[:, s], np.array(pred)[:, s], pos_label=None)
                au = auc(sp, se)
                sens.append(se[best_ind])
                spec.append(sp[best_ind])
                thre.append(th[best_ind])
                aucc.append(au)
            return sens, spec, thre, aucc
        else:
            for s, cls in enumerate(self.classes):
                sp, se, th, best_ind = spec_sens(np.array(true)[:, s], np.array(pred)[:, s], pos_label=None)
                au = auc(sp, se)
                sens.append(se[best_ind])
                spec.append(sp[best_ind])
                thre.append(th[best_ind])
                aucc.append(au)
            return sens, spec, thre, aucc

    def gen_heatmap(self, train_loader, loop=None, k=1):
        print('heatmap')
        grad_cam = GradCam(model=self.net, target_layer_names=["layer4"], use_cuda=True)
        self.net.eval()
        for batch_idx, (inputs, labels, years, paths) in enumerate(train_loader):
            inputs = Variable(inputs.cuda())
            # generate heatmap
            imgs = inputs.data.cpu().numpy()
            for param in self.net.parameters():
                param.requires_grad = True
            for s, img in enumerate(tqdm(imgs)):
                mask = grad_cam(inputs[s].unsqueeze(0), None)
                img = img.transpose([1, 2, 0])
                img = img - np.min(img.flatten())
                img = img / np.max(img.flatten())
                cam = show_cam_on_image(img, mask)
                try:
                    shutil.copy(paths[s], paths[s].replace('tongji_breast_patches', 'tongji_breast_patches_heatmap'))
                    cv2.imwrite(paths[s].replace('tongji_breast_patches', 'tongji_breast_patches_heatmap')
                                .replace('.png', '_loop%d_k%d_cam.png' % (loop, k)), np.uint8(255 * cam))
                except:
                    print(paths[s].replace('tongji_breast_patches', 'tongji_breast_patches_heatmap'))
            for param in self.net.parameters():
                param.requires_grad = False

    # calculate specificity and sensitivity
    def abandon_with_confidence(self, train_loader, num, type='train', loop=None, k=1, th=0.5):
        print('abandon_with_confidence')
        # grad_cam = GradCam(model=self.net, target_layer_names=["layer4"], use_cuda=True)
        self.net.eval()
        for param in self.net.parameters():
            param.requires_grad = False
        all_pred = []
        all_label = []
        all_path = []
        all_single_im = {}
        for batch_idx, (inputs, labels, years, paths) in enumerate(train_loader):
            inputs = Variable(inputs.cuda())
            # # generate heatmap
            # imgs = inputs.data.cpu().numpy()
            # for param in self.net.parameters():
            #     param.requires_grad = True
            # for k, img in enumerate(tqdm(imgs)):
            #     mask = grad_cam(inputs[k].unsqueeze(0), None)
            #     img = img.transpose([1, 2, 0])
            #     img = img - np.min(img.flatten())
            #     img = img / np.max(img.flatten())
            #     cam = show_cam_on_image(img, mask)
            #     try:
            #         shutil.copy(paths[k], paths[k].replace('tongji_breast_patches', 'tongji_breast_patches_heatmap'))
            #         cv2.imwrite(paths[k].replace('tongji_breast_patches', 'tongji_breast_patches_heatmap')
            #                 .replace('.png', '_cam.png'), np.uint8(255 * cam))
            #     except:
            #         print(paths[k].replace('tongji_breast_patches', 'tongji_breast_patches_heatmap'))
            #
            # for param in self.net.parameters():
            #     param.requires_grad = False

            preds = self.net(inputs)
            # pred = F.softmax(preds, dim=1).data.cpu().numpy()
            pred = F.sigmoid(preds).data.cpu().numpy()
            pred = list(pred[:, 0])
            labels = labels.data.cpu().numpy()
            for k, p in enumerate(paths):
                name = p.split('/')[-1].split('_')[0]
                if not name in all_single_im.keys():
                    all_single_im[name] = [(pred[k], labels[k, 0], p)]
                else:
                    all_single_im[name].append((pred[k], labels[k, 0], p))

            all_pred.extend(pred)
            all_label.extend(list(labels[:, 0]))
            all_path.extend(list(paths))
            print(batch_idx, num, end='\r')
        print()

        all_path = np.array(all_path)
        all_pred = np.array(all_pred)
        all_label = np.array(all_label)
        th1, th2, spec, sens, thres, best_thres = self.thres_spec_sens(all_pred, all_label, all_path, th)
        low_conf = self.save_spec_sens(all_path, all_pred, th1, th2, spec, sens, thres, best_thres, loop, k)
        self.save_spec_sens_single(all_single_im, best_thres, loop, k)
        return low_conf

    def thres_spec_sens(self, all_pred, all_label, all_path, th=0.5):
        # patches auc
        all_pred = np.array(all_pred)
        all_label = np.array(all_label)
        all_path = np.array(all_path)

        spec, sens, thres, best_index = spec_sens(all_label, all_pred, pos_label=None)
        best_index = np.argmin(np.abs(np.array(spec) - np.array(sens)))
        best_thres = thres[best_index]
        th1, th2 = 0, 0
        for s in range(100):
            s_ = s * 0.001
            th1 = thres[best_index] - s_
            th2 = thres[best_index] + s_
            n1 = len(all_pred[all_pred < th1])
            n2 = len(all_pred[all_pred > th2])
            if (n1 + n2) * 1.0 / len(all_pred) < 0.7:
                break
        #
        # th1 = thres[best_index] - 0.05
        # th2 = thres[best_index] + 0.05

        # ind_spec = np.argmin(np.abs(spec-th))
        # ind_sens = np.argmin(np.abs(sens-th))
        # th1 = thres[ind_spec]
        # th2 = thres[ind_sens]
        return th1, th2, spec, sens, thres, best_thres

    def save_spec_sens(self, all_path, all_pred, th1, th2, spec, sens, thres, best_thres, loop, k):
        spec = [1.0] + list(spec) + [0.0]
        sens = [0.0] + list(sens) + [1.0]
        thres = [1.0] + list(thres) + [0.0]
        aucc = 0
        try:
            aucc = auc(spec, sens)
            print('th1: %.3f, th2: %.3f, auc: %.3f' % (th1, th2, aucc))
        except:
            print('no auc')
        plt.plot(np.array(spec), np.array(sens), color='r', alpha=0.7)
        plt.plot(np.array(thres), np.array(sens), color='b', alpha=0.7)
        plt.plot(np.array(thres), np.array(spec), color='g', alpha=0.7)
        plt.plot([th1, th1], [0.0, 1.0], color='y', alpha=0.7)
        plt.plot([th2, th2], [0.0, 1.0], color='c', alpha=0.7)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.title('%.3f, %.3f' % (th1, th2))
        plt.savefig('fig/single_im_%s_loop%d_k%d_auc%.3f_bestThes%.3f.jpg' % (type, loop, k, aucc, best_thres))
        plt.close()
        #
        low_conf = list(all_path[(all_pred >= th1) & (all_pred <= th2)])
        return low_conf
        # for lc in tqdm(low_conf):
        #     move(lc, lc.replace('tongji_breast_patches/', 'tongji_breast_patches_confidence_classification/low_conf/'))
        # print('throw %d samples' % len(low_conf))

    def save_spec_sens_single(self, all_single_im, best_thres, loop, k):
        # single image auc
        y_pred, y_pred_max, y_true = [], [], []
        for na in all_single_im.keys():
            infos = all_single_im[na]
            label = str(infos[0][1])
            if not os.path.exists('/media/data_storage/HE/collect_patch_224/tmp_result/' + na + '_' + label):
                os.mkdir('/media/data_storage/HE/collect_patch_224/tmp_result/' + na + '_' + label)
            preds = np.array([m[0] for m in infos])
            paths = [m[2] for m in infos]
            for p in paths:
                shutil.copy(p, p.replace('tongji_breast_patches', 'tmp_result/' + na + '_' + label))
            # print(preds)

            # preds[preds >= best_thres] = 1.0
            # preds[preds < best_thres] = 0.0
            mean_pred = np.mean(np.array(preds))
            max_pred = np.max(preds)

            # print(preds.reshape([1,-1]), mean_pred, infos[0][1])
            y_pred.append(mean_pred)
            y_true.append(infos[0][1])
            y_pred_max.append(max_pred)

        y_pred = np.array(y_pred)
        y_pred_max = np.array(y_pred_max)
        y_true = np.array(y_true)
        spec, sens, thres, best_index = spec_sens(y_true, y_pred)
        spec_max, sens_max, thres_max, best_index_max = spec_sens(y_true, y_pred_max)
        best_thres = thres[best_index]

        spec = [1.0] + list(spec) + [0.0]
        sens = [0.0] + list(sens) + [1.0]
        # thres = [1.0] + list(thres) + [0.0]

        spec_max = [1.0] + list(spec_max) + [0.0]
        sens_max = [0.0] + list(sens_max) + [1.0]
        # thres_max = [1.0] + list(thres_max) + [0.0]

        aucc_mean = 0
        # aucc_max = 0
        try:
            aucc_mean = auc(spec, sens)
            aucc_max = auc(spec_max, sens_max)
            print('single image auc: mean-%.3f, max-%.3f' % (aucc_mean, aucc_max))
        except:
            print('no single auc')
        plt.plot(np.array(spec), np.array(sens), color='r', alpha=0.7)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.savefig('fig/single_im_%s_loop%d_k%d_auc%.3f_bestThes%.3f.jpg' % (type, loop, k, aucc_mean, best_thres))
        plt.close()

    def _train_val(self, epoch, data_loader, num, optimizer_ft=None, type='train'):
        ave_loss = []
        sens, spes, accs = {}, {}, {}
        ave_sens, ave_spes, ave_accs = {}, {}, {}

        if type == 'predict':
            predict_dict = {}
            predict_label_dict = {}
            predict_middle_vector = {}

        for batch_idx, (inputs, labels, paths) in enumerate(data_loader):
            # print('batch_idx', batch_idx)
            inputs = Variable(inputs.cuda())
            if self.multi_layer:
                labels = [Variable(m.cuda()) for m in labels]
            else:
                labels = Variable(labels.cuda())
            if type == 'train':
                optimizer_ft.zero_grad()

            if len(inputs.size()) == 5:
                bs, ncrops, c, h, w = inputs.size()
                preds_n = self.net(inputs.view(-1, c, h, w))
                preds = preds_n.view(bs, ncrops, -1).mean(1)
            else:
                preds = self.net(inputs)
                # print(preds)
                if type == 'predict':
                    middle_v = self.net.pool(self.net.features(inputs))
                    middle_v = middle_v.cpu().numpy().squeeze()
                    # print(middle_v.shape)
            # print(preds.shape)
            # exit()
            if self.multi_task:
                loss = 0
                for s, name in enumerate(self.classes):
                    # 是否属于当前类别的pred,
                    pred = preds[s]
                    # 如果是当前类，那么label为1, 但是相当于argmax([1, 0]) 为0， 所以用1减
                    # labels [16, 6, 2]
                    label = torch.argmax(labels[:, s, :], dim=1)
                    loss += self.criterion(pred, label.long())
                label = labels.data.cpu().numpy()
                # label [16, 6] 选择第0个位置的为当前类的概率
                label = label[:, :, 0]

                pred = np.zeros(labels.shape[:2])
                for s, name in enumerate(self.classes):
                    p = preds[s]
                    # p = F.softmax(p, dim=1).data.cpu().numpy()[:, 0]
                    p = F.sigmoid(p).data.cpu().numpy()[:, 0]
                    pred[:, s] = p
            else:
                # 非multi-class的 multi_layer方式
                if self.multi_layer:
                    loss = 0
                    pred = []
                    for k, p in enumerate(preds):
                        # print('2', p)
                        # print('3', labels[k])
                        loss += self.criterion[k](p, torch.argmax(labels[k], dim=1))
                        # print(loss)
                        # pred.append(F.softmax(p, dim=1).data.cpu().numpy())
                        pred.append(F.sigmoid(p).data.cpu().numpy())
                    label = [m.data.cpu().numpy() for m in labels]
                # 非multi-class的 非multi_layer方式
                else:
                    # 这时的labels为[16, 6]
                    loss = self.criterion(preds, torch.argmax(labels, dim=1))
                    label = labels.data.cpu().numpy()
                    # pred = F.softmax(preds, dim=1).data.cpu().numpy()
                    pred = F.sigmoid(preds).data.cpu().numpy()

            # if self.multi_task:
            #     pred = np.zeros(labels.shape[:2])
            #     for s, name in enumerate(self.classes):
            #         p = preds[s]
            #         p = F.softmax(p, dim=1).data.cpu().numpy()[:, 0]
            #         pred[:, s] = p
            # else:
            #     # pred: [batch_size, num_class]
            #     pred = F.softmax(preds, dim=1).data.cpu().numpy()

            if type == 'predict':
                if self.multi_layer:
                    for s, p in enumerate(pred[-1]):
                        cur_name = paths[s]
                        while cur_name in predict_dict.keys():
                            cur_name = cur_name + '_fake'
                        predict_dict[cur_name] = [m[s] for m in pred]
                        predict_middle_vector[cur_name] = middle_v[s]
                        predict_label_dict[cur_name] = [m[s] for m in label]
                else:
                    for s, p in enumerate(pred):
                        cur_name = paths[s]
                        while cur_name in predict_dict.keys():
                            cur_name = cur_name + '_fake'
                        predict_dict[cur_name] = p
                        predict_middle_vector[cur_name] = middle_v[s]
                        predict_label_dict[cur_name] = label[s]

                print('%d/%d' % (batch_idx, num), end='\r')
            else:
                if type == 'train':
                    loss.backward()
                    optimizer_ft.step()
                # print('after backward')
                loss = loss.data.cpu().numpy()
                ave_loss.append(loss)
                ave_loss_ = np.mean(ave_loss)

                if self.multi_layer:
                    str_sens, str_spes, str_accs = '', '', ''
                    for k, single_layer_classes in enumerate(self.multi_layer_classes):
                        tmp_sens, tmp_spes, tmp_accs = {}, {}, {}
                        for s, cls in enumerate(single_layer_classes):
                            new_cls = 'layer%d_' % k + cls
                            if not new_cls in sens.keys():
                                sens[new_cls], spes[new_cls], accs[new_cls] = [0, 0], [0, 0], [0, 0]
                            # 遍历batch中的每一个pred
                            for m, p in enumerate(pred[k]):
                                if np.argmax(p) == s:
                                    sens[new_cls][0] += label[k][m][s]
                                    accs[new_cls][0] += label[k][m][s]
                                else:
                                    spes[new_cls][0] += np.sum(label[k][m]) - label[k][m][s]
                                    accs[new_cls][0] += np.sum(label[k][m]) - label[k][m][s]

                            sens[new_cls][1] = sens[new_cls][1] + np.sum(label[k][:, s])
                            spes[new_cls][1] = spes[new_cls][1] + len(label[k]) - np.sum(label[k][:, s])
                            accs[new_cls][1] += len(label[k])

                            tmp_sens[new_cls] = '%.3f' % (sens[new_cls][0] / (sens[new_cls][1] + 0.001))
                            tmp_spes[new_cls] = '%.3f' % (spes[new_cls][0] / (spes[new_cls][1] + 0.001))
                            tmp_accs[new_cls] = '%.3f' % (accs[new_cls][0] / (accs[new_cls][1] + 0.001))

                        str_sens = str_sens + 'l%d ' % k + ','.join([tmp_sens[m] for m in tmp_sens.keys()]) + '-'
                        str_spes = str_spes + 'l%d ' % k + ','.join([tmp_spes[m] for m in tmp_spes.keys()]) + '-'
                        str_accs = str_accs + 'l%d ' % k + ','.join([tmp_accs[m] for m in tmp_accs.keys()]) + '-'

                    print('epoch%d-%d/%d |t_loss: %.3f |avg_loss: %.3f' % (epoch, batch_idx, num, loss, ave_loss_),
                          '|sens:', str_sens, '|spes:', str_spes, '|accs:', str_accs, end='\r')

                else:
                    for s, new_cls in enumerate(self.classes):
                        if not new_cls in sens.keys():
                            sens[new_cls], spes[new_cls], accs[new_cls] = [0, 0], [0, 0], [0, 0]
                        # 遍历batch中的每一个pred
                        for m, p in enumerate(pred):
                            if np.argmax(p) == s:
                                sens[new_cls][0] += label[m][s]
                                accs[new_cls][0] += label[m][s]
                            else:
                                spes[new_cls][0] += np.sum(label[m]) - label[m][s]
                                accs[new_cls][0] += np.sum(label[m]) - label[m][s]

                        sens[new_cls][1] = sens[new_cls][1] + np.sum(label[:, s])
                        spes[new_cls][1] = spes[new_cls][1] + len(label) - np.sum(label[:, s])
                        accs[new_cls][1] += len(label)

                        ave_sens[new_cls] = '%.3f' % (sens[new_cls][0] / (sens[new_cls][1] + 0.001))
                        ave_spes[new_cls] = '%.3f' % (spes[new_cls][0] / (spes[new_cls][1] + 0.001))
                        ave_accs[new_cls] = '%.3f' % (accs[new_cls][0] / (accs[new_cls][1] + 0.001))

                    str_sens = ','.join([ave_sens[m] for m in ave_sens.keys()])
                    str_spes = ','.join([ave_spes[m] for m in ave_spes.keys()])
                    str_accs = ','.join([ave_accs[m] for m in ave_accs.keys()])
                    print('epoch%d-%d/%d |t_loss: %.3f |avg_loss: %.3f' % (epoch, batch_idx, num, loss, ave_loss_),
                          '|sens:', str_sens, '|spes:', str_spes, '|accs:', str_accs, end='\r')

        torch.cuda.empty_cache()
        if type == 'predict':
            return predict_dict, predict_label_dict, predict_middle_vector
        return ave_loss


class BranchNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BranchNet, self).__init__()
        branch = [  # BasicConv2d(in_channels=in_channels, out_channels=out_channels),
            BasicConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3)]
        self.branch = nn.Sequential(*branch)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        return self.branch(x)


class ClassifierNet(nn.Module):
    def __init__(self, in_channels, num_class, pool_size=8):
        super(ClassifierNet, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=pool_size)
        classifier = [
            nn.Dropout(0.5),
            nn.Linear(in_features=in_channels, out_features=num_class)]
        self.classifier = nn.Sequential(*classifier)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class RootResnet50(nn.Module):
    def __init__(self, pool_size, pretrained=True):
        super(RootResnet50, self).__init__()
        self.pretrained = pretrained
        self.model = self.resnet_50_model(pool_size=pool_size, num_classes=1000)
        self.features = self.model.features
        # if self.multi_task:
        #     self.classifier = []
        #     for cls in range(num_classes):
        #         self.classifier.append(BranchNet(branch_input=self.model._classifier.in_features, num_class=2))
        # else:
        #     self.classifier = BranchNet(branch_input=self.model._classifier.in_features, num_class=num_classes)

    def resnet_50_model(self, pool_size, num_classes):
        # transform_input 是用来做transform的，如果已经做过就可以不用做了
        self.pool = nn.AvgPool2d(kernel_size=pool_size)
        original_model = torchvision_models.resnet50(pretrained=self.pretrained)
        finetune_model = make_model('resnet50', num_classes=num_classes, pool=self.pool, pretrained=self.pretrained)
        self.copy_module_weights(original_model.fc, finetune_model._classifier)
        # assert_equal_outputs(input_var, original_model, finetune_model)
        return finetune_model

    def copy_module_weights(self, from_module, to_module):
        to_module.weight.data.copy_(from_module.weight.data)
        to_module.bias.data.copy_(from_module.bias.data)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        # x = x.view(x.size(0), -1)
        # if self.multi_task:
        #     return [m(x) for m in self.classifier]
        # else:
        #     x = self.classifier(x)
        #     return x
        return x


class RootDensenet(nn.Module):
    def __init__(self, pool_size, pretrained=True):
        super(RootDensenet, self).__init__()
        self.pretrained = pretrained
        self.model = self.densenet_model(pool_size=pool_size, num_classes=1000)
        self.features = self.model.features
        # if self.multi_task:
        #     self.classifier = []
        #     for cls in range(num_classes):
        #         self.classifier.append(BranchNet(branch_input=self.model._classifier.in_features, num_class=2))
        # else:
        #     self.classifier = BranchNet(branch_input=self.model._classifier.in_features, num_class=num_classes)

    def densenet_model(self, pool_size, num_classes):
        # transform_input 是用来做transform的，如果已经做过就可以不用做了
        self.pool = nn.AvgPool2d(kernel_size=pool_size)
        original_model = torchvision_models.densenet121(pretrained=self.pretrained)
        finetune_model = make_model('densenet121', num_classes=num_classes, pool=self.pool, pretrained=self.pretrained)
        # self.copy_module_weights(original_model.classifier, finetune_model._classifier)
        # assert_equal_outputs(input_var, original_model, finetune_model)
        return finetune_model

    def copy_module_weights(self, from_module, to_module):
        to_module.weight.data.copy_(from_module.weight.data)
        to_module.bias.data.copy_(from_module.bias.data)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        # x = x.view(x.size(0), -1)
        # if self.multi_task:
        #     return [m(x) for m in self.classifier]
        # else:
        #     x = self.classifier(x)
        #     return x
        return x

class RootVgg16(nn.Module):
    def __init__(self, pool_size, pretrained=True):
        super(RootVgg16, self).__init__()
        self.pretrained = pretrained
        self.model = self.vgg16_model(pool_size=pool_size, num_classes=1000)
        self.features = self.model.features
        # if self.multi_task:
        #     self.classifier = []
        #     for cls in range(num_classes):
        #         self.classifier.append(BranchNet(branch_input=self.model._classifier.in_features, num_class=2))
        # else:
        #     self.classifier = BranchNet(branch_input=self.model._classifier.in_features, num_class=num_classes)

    def vgg16_model(self, pool_size, num_classes):
        # transform_input 是用来做transform的，如果已经做过就可以不用做了
        self.pool = nn.AvgPool2d(kernel_size=pool_size)
        original_model = torchvision_models.vgg16_bn(pretrained=self.pretrained)
        finetune_model = make_model('vgg16_bn', num_classes=num_classes, pool=self.pool, pretrained=self.pretrained, input_size=(224, 224))
        # self.copy_module_weights(original_model.classifier, finetune_model._classifier)
        # assert_equal_outputs(input_var, original_model, finetune_model)
        return finetune_model

    def copy_module_weights(self, from_module, to_module):
        to_module.weight.data.copy_(from_module.weight.data)
        to_module.bias.data.copy_(from_module.bias.data)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        # x = x.view(x.size(0), -1)
        # if self.multi_task:
        #     return [m(x) for m in self.classifier]
        # else:
        #     x = self.classifier(x)
        #     return x
        return x

class RootInceptionV3(nn.Module):
    def __init__(self, pool_size, pretrained=True):
        super(RootInceptionV3, self).__init__()
        self.pretrained = pretrained
        self.model = self.inception_v3_model(pool_size=pool_size, num_classes=1000)
        self.features = self.model.features
        # if self.multi_task:
        #     self.classifier = []
        #     for cls in range(num_classes):
        #         self.classifier.append(BranchNet(branch_input=self.model._classifier.in_features, num_class=2))
        # else:
        #     self.classifier = BranchNet(branch_input=self.model._classifier.in_features, num_class=num_classes)

    def inception_v3_model(self, pool_size, num_classes):
        # transform_input 是用来做transform的，如果已经做过就可以不用做了
        self.pool = nn.AvgPool2d(kernel_size=pool_size)
        original_model = torchvision_models.inception_v3(pretrained=self.pretrained, transform_input=False)
        finetune_model = make_model('inception_v3', num_classes=num_classes, pool=self.pool, pretrained=self.pretrained)
        self.copy_module_weights(original_model.fc, finetune_model._classifier)
        # assert_equal_outputs(input_var, original_model, finetune_model)
        return finetune_model

    def copy_module_weights(self, from_module, to_module):
        to_module.weight.data.copy_(from_module.weight.data)
        to_module.bias.data.copy_(from_module.bias.data)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        # x = x.view(x.size(0), -1)
        # if self.multi_task:
        #     return [m(x) for m in self.classifier]
        # else:
        #     x = self.classifier(x)
        #     return x
        return x


class HERootResnet(models.resnet.ResNet):
    def __init__(self, avgpool_size=7):
        super(HERootResnet, self).__init__(Bottleneck, [3, 4, 6, 3], num_classes=1000)
        self.avgpool = nn.AvgPool2d(kernel_size=avgpool_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # print(x.shape)
        x = self.avgpool(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        return x


class HEPatchNet(nn.Module):
    def __init__(self, classifier_input, num_class, size):
        super(HEPatchNet, self).__init__()
        self.he_root = HERootResnet(avgpool_size=int(size / 32))
        self.classifier1 = BranchNet(branch_input=classifier_input, num_class=num_class)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def freeze_bn(self):
        # Freeze BatchNorm layers.
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm3d):
                layer.eval()

    def forward(self, x):
        x = self.he_root(x)
        # print(x1.shape, x2.shape, x3.shape)
        # s = torch.cat([x1, x2, x3], dim=1)
        # print(s.shape)
        s = self.classifier1(x)
        # s2 = self.regression(s)
        return s

##############################################################################################
#  loss function
class FocalLoss(nn.Module):
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def focalloss(self, inputs, targets):
        # set_trace()
        N = inputs.size(0)
        C = inputs.size(1)
        # P = F.softmax(inputs, dim=1)
        P = F.sigmoid(inputs)
        # print('1', P)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        class_mask1 = inputs.data.new(N, C).fill_(1)
        class_mask1 = Variable(class_mask1)

        # targets = torch.argmax(targets, 1)
        # print(targets)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        class_mask = class_mask.cuda()
        class_mask1.scatter_(1, ids.data, 0.)

        # probs = (P * class_mask).sum(1).view(-1, 1)
        probs = P * class_mask + (1-P) * class_mask1
        # print('2', probs)

        log_p = probs.log()
        # print('probs size= {}'.format(probs.size()))
        # print(probs)

        batch_loss = -(torch.pow((1 - probs), self.gamma)) * log_p
        # print('-----bacth_loss------')
        # print(batch_loss)
        loss = batch_loss.mean()
        # if self.size_average:
        #     loss = batch_loss.mean()
        # else:
        #     loss = batch_loss.sum()
        return loss

    def forward(self, top_pred, top_tar):
        loss1 = self.focalloss(top_pred, top_tar)
        return loss1
