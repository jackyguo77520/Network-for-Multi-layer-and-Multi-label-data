
import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
n_folds = 5
epochs = 40
batch_size = 16
size = 299
num_workers = 40
times_per_epoch = 7
epoch_th = np.array([5, 20])
# day = '20190110'
# day = '20190212'
day = '20190215'
multi_task = False
fore = False
val = True
multi_layer = True

s1 = 'multi_task' if multi_task else 'multi_class'
s2 = 'val' if val else 'test'
s3 = 'fore' if fore else 'back'
s4 = 'multi_layer' if multi_layer else 'single_layer'

# 'eye_fore_multi_task_multi_layer'
model_name = 'eye_%s_%s_%s' % (s3, s1, s4)
# '17class_multi_task_multi_layer_val_fore'
save_dir = '17class_%s_%s_%s_%s' % (s1, s4, s2, s3)
# pickle save file
save_file = '17class_%s_%s_%s_%s.pkl' % (s1, s4, s2, s3)

train_root = '/media/data_storage/ophthalmology/class_statistics/'

# multi layer classes fore
classes_layer0_fore = ['Cataract', 'Ocular_Surface', 'Ocular_Neoplasma', 'Normal_Surface']
classes_layer1_fore = ['Cataract', 'Conjunctivitis', 'Cornea_Degeneration', 'Cornea_Infectious', 'Cornea_Neoplasm',
                       'Cornea_Non_Infectious', 'New_Creature', 'Scleritis', 'Normal_Surface']
multi_layer_dict_fore = {'Cataract': ['Cataract'],
                        'Conjunctivitis': ['Ocular_Surface'],
                        'Cornea_Degeneration':['Ocular_Surface'],
                        'Cornea_Infectious': ['Ocular_Surface'],
                        'Cornea_Non_Infectious': ['Ocular_Surface'],
                        'Scleritis': ['Ocular_Surface'],
                        'Cornea_Neoplasm': ['Ocular_Neoplasma'],
                        'New_Creature': ['Ocular_Neoplasma'],
                        'Normal_Surface': ['Normal_Surface']}
classes_fore = [classes_layer0_fore, classes_layer1_fore]

# multi layer classes back
classes_layer0_back = ['Glaucoma', 'Vitreous_Retinal', 'Normal_Fundus']
classes_layer1_back = ['Glaucoma', 'Macular', 'Optic_Nerve', 'Refractive', 'Retinal_Degeneration', 'Retinal_Detachment',
                       'Retinal_Vascular', 'Normal_Fundus']
multi_layer_dict_back = {'Glaucoma': ['Glaucoma'],
                         'Macular': ['Vitreous_Retinal'],
                         'Optic_Nerve': ['Vitreous_Retinal'],
                         'Refractive': ['Vitreous_Retinal'],
                         'Retinal_Degeneration': ['Vitreous_Retinal'],
                         'Retinal_Detachment': ['Vitreous_Retinal'],
                         'Retinal_Vascular': ['Vitreous_Retinal'],
                         'Normal_Fundus': ['Normal_Fundus']}
classes_back = [classes_layer0_back, classes_layer1_back]

if fore:
    classes_last = classes_fore[-1] if multi_layer else classes_fore
    classes_all = classes_fore
    multi_layer_dict = multi_layer_dict_fore
else:
    classes_last = classes_back[-1] if multi_layer else classes_back
    classes_all = classes_back
    multi_layer_dict = multi_layer_dict_back


