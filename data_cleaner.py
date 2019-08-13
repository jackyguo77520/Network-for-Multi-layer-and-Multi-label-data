from glob import glob
import os
from PIL import Image
import numpy as np
import shutil
import pickle
from multiprocessing import Pool


def run_get_hash_of_image(p):
    """
    该函数通过读取图像p，获得其中间的一条数据，然后对其进行hash编码
    :param p:
    :return:
    """
    img = Image.open(p).convert("RGB")
    img = np.array(img)
    h, w, c = img.shape
    g = img[:, int(w/2), 0]
    v = ''
    for gs in g:
        v += str(gs)
    # v = str(img[:, int(w/2), 0])
    hash_v = str(hash(v))
    return hash_v


def multi_thread_get_hash_of_image(ps):
    pool = Pool()
    hash_vs = pool.map(run_get_hash_of_image, ps)
    pool.close()
    pool.join()
    return hash_vs


def delete_same_image_in_one_folder(path, abandon_path):
    """
    将path文件夹下，相同的样本挑选出来，留一份，多余的放入abandon_path中
    并且会在abandon_path下面自动生成和path的最次级文件夹相同名字的文件夹
    :param path:
    :param abandon_path:
    :return:
    """
    for folder in glob(os.path.join(path, '*')):
        name = folder.split('/')[-1]
        abandon_folder = abandon_path + name + '/'

        if not os.path.exists(abandon_folder):
            os.mkdir(abandon_folder)

        ps = glob(os.path.join(folder, '*'))
        print(name, len(ps))
        ss = {}
        total = 0
        hash_vs = multi_thread_get_hash_of_image(ps)
        for k in range(len(ps)):
            hash_v = hash_vs[k]
            p = ps[k]
            if not hash_v in ss.keys():
                ss[hash_v] = [p]
            else:
                ss[hash_v] += [p]
                shutil.move(p, abandon_folder+p.split('/')[-1])
                total += 1

        ps = glob(os.path.join(folder, '*'))
        print(name, len(ps))


def get_image_label_over_all_folder(root_folder, abandon_folder=None, save_path=None, load=True):
    """
    该函数用来寻找类别间的相同图像，以找到某些图像属于多个类别的情况
    最终生成以图像名为key，以类别list为value的dict
    首先会去掉一个文件夹中相同的数据
    :param root_folder: 为包含多个类别folders的文件夹
    :param abandon_folder:
    :param save_path:
    :param load:
    :return:
    """
    hash_ps = {}
    hash_class = {}

    # 如果存在save_path并且load为True
    if load and os.path.exists(save_path):
        with open(save_path, 'rb') as f:
            ps_class = pickle.load(f)
    else:
        # 遍历root_folder下的每个类文件夹
        for folder in glob(os.path.join(root_folder, '*')):
            name = folder.split('/')[-1]
            ps = glob(os.path.join(folder, '*'))
            hash_vs = multi_thread_get_hash_of_image(ps)

            # 去掉一个类文件夹中重复的样本
            new_hash_vs = []
            new_ps = []
            tmp_dic = {}
            for k in range(len(ps)):
                hash_v = hash_vs[k]
                p = ps[k]
                if not hash_v in tmp_dic.keys():
                    tmp_dic[hash_v] = [p]
                    new_hash_vs.append(hash_v)
                    new_ps.append(p)
                else:
                    pass
                    # shutil.move(p, abandon_folder+p.split('/')[-1])
            print('class %s, 在去重前有%d样本，去重后有%d样本' % (name, len(ps), len(new_ps)))

            # 统计一个样本可以分属于多个类文件夹
            for k in range(len(new_ps)):
                hash_v = new_hash_vs[k]
                p = new_ps[k]
                if not hash_v in hash_ps.keys():
                    hash_ps[hash_v] = [p]
                    hash_class[hash_v] = [name]
                else:
                    hash_ps[hash_v].append(p)
                    hash_class[hash_v].append(name)

        # 统计多个类别样本的个数
        multi_label_sta = {}
        for key in hash_class.keys():
            num = len(hash_class[key])
            if not num in multi_label_sta.keys():
                multi_label_sta[num] = 1
            else:
                multi_label_sta[num] += 1
        print('统计多类别样本的个数：', multi_label_sta)

        # 将hash_ps和hash_class综合变成key为图像名称，value为class的list
        ps_class = {}
        for key in hash_ps.keys():
            ps = hash_ps[key]
            for p in ps:
                ps_class[p] = hash_class[key]

        # 好不容易计算出来的保存一下
        with open(save_path, 'wb') as f:
            pickle.dump(ps_class, f)

    return ps_class
