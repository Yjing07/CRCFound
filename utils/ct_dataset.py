import os
import SimpleITK as sitk
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset,DataLoader
# from utils.util import resize_image_itk,randomcrop,crop,read_json #utils.
import random
random.seed(42)
import joblib
from torch.utils.data import BatchSampler

def window_level_normalization(img, level=35, window=300):
    min_HU = level - window / 2
    max_HU = level + window / 2
    img[img > max_HU] = max_HU
    img[img < min_HU] = min_HU
    img = 1. * (img - min_HU) / (max_HU - min_HU)
    return img

def process(img, level=35, window=300):
    img = window_level_normalization(img, level=35, window=300)
    return img[:,50:450]*255

class colon_cancer(Dataset):
    """自定义数据集"""

    def __init__(self, train_ind, data_path, lab_dict, shape):
        super(colon_cancer,self).__init__()

        self.images_ind = train_ind
        self.lab_dict = lab_dict
        self.ct_path = data_path
        self.size = shape
        self.tuominID = [self.lab_dict[i]['tuominID'] for i in self.images_ind]
        self.label = [self.lab_dict[i]['dfs.event'] for i in self.images_ind]
        self.delay = [self.lab_dict[i]['dfs.delay'] for i in self.images_ind]
        self.ct_texts = [self.lab_dict[i]['ct_report'] for i in self.images_ind]
        self.wsi_texts = [self.lab_dict[i]['patho_report'] for i in self.images_ind]

    def __len__(self):
        return len(self.tuominID)

    def __getitem__(self, i):
        img_root = os.path.join(self.ct_path,'image',self.tuominID[i])
        roi_root = os.path.join(self.ct_path,'roi',self.tuominID[i])
        img = sitk.ReadImage(img_root)
        roi = sitk.ReadImage(roi_root)

        img = sitk.GetArrayFromImage(img).astype(np.float32)
        roi = sitk.GetArrayFromImage(roi).astype(np.float32)
        # img,roi = randomcrop(img,roi,self.size)
        img = torch.from_numpy(img)
        roi = torch.from_numpy(roi)
        
        event =  torch.tensor(self.label[i])
        delay =  torch.tensor(self.delay[i])
        text_caption = self.ct_texts[i][:500]
        wsi_caption = self.wsi_texts[i][:500]
        
        return img,roi, event, delay, text_caption, wsi_caption, self.images_ind[i]   #,os.path.basename(self.images_ind[i])


def randomcrop(img,roi,size):
    c,h,w =img.shape
    cc,hh,ww = size
    ci=random.randint(0,c-cc)
    hi=random.randint(0,h-hh)
    wi=random.randint(0,w-ww)
    # return img[ci:ci+cc],roi[ci:ci+cc]
    if roi[ci:ci+cc,hi:hi+hh,wi:wi+ww].sum():
        return img[ci:ci+cc,hi:hi+hh,wi:wi+ww],roi[ci:ci+cc,hi:hi+hh,wi:wi+ww]
    else:
        return img[:cc,hi:hi+hh,wi:wi+ww],roi[:cc,hi:hi+hh,wi:wi+ww]
    
class pretrain_dataset(Dataset):
    """自定义数据集"""

    def __init__(self, ct_data_path, lab_dict, size):
        super(pretrain_dataset,self).__init__()
        self.ct_path = ct_data_path
        # self.images_ind = list(lab_dict.keys())
        self.images_ind = list(lab_dict)
        self.size = size

    def __len__(self):
        return len(self.images_ind)

    def __getitem__(self, i):
        img_root = os.path.join(self.ct_path,'image',self.images_ind[i])
        # roi_root = os.path.join(self.ct_path,'roi',self.images_ind[i])
        # print(img_root)
        img = sitk.ReadImage(img_root)
        # roi = sitk.ReadImage(roi_root)

        img = sitk.GetArrayFromImage(img).astype(np.float32)
        # roi = sitk.GetArrayFromImage(roi).astype(np.float32)
        # if img.shape[-1] < 320:
        #     print(img_root)
        # img,roi = randomcrop(img,roi,self.size)
        img = torch.from_numpy(img)
        # roi = torch.from_numpy(roi)

        # texts = "一张位于{}的{}ct图像。".format(self.tumor_location[i], self.histopathology[i])
        return img, self.images_ind[i].split('.')[0]

class MAE_class(Dataset):
    """自定义数据集"""

    def __init__(self, img_ind, ct_data_path, lab_dict, size):
        super(MAE_class,self).__init__()

        self.img_list = img_ind
        self.images_idx = list(self.img_list.keys())
        self.ct_path = ct_data_path
        self.all_label = lab_dict
        self.size = size

    def __len__(self):
        return len(self.images_idx)
    
    def __getitem__(self, i):

        tumormin_id = self.images_idx[i]
        img_root = os.path.join(self.ct_path,'image', tumormin_id)
        roi_root = os.path.join(self.ct_path,'roi', tumormin_id)
        img = sitk.ReadImage(img_root)
        roi = sitk.ReadImage(roi_root)

        img = sitk.GetArrayFromImage(img).astype(np.float32)
        roi = sitk.GetArrayFromImage(roi).astype(np.float32)
        img = torch.from_numpy(img)
        roi = torch.from_numpy(roi)

        label = self.img_list[tumormin_id]
        
        return img,roi, label,tumormin_id

class MAE_fusion_class(Dataset):
    """自定义数据集"""

    def __init__(self, img_ind, ct_data_path, lab_dict, size):
        super(MAE_fusion_class,self).__init__()

        self.img_list = img_ind
        self.images_idx = list(self.img_list.keys())
        self.ct_path = ct_data_path
        self.all_label = lab_dict
        self.size = size
        self.text_embeddings = joblib.load('report_features.pkl')

    def __len__(self):
        return len(self.images_idx)
    
    def __getitem__(self, i):
        tumormin_id = self.images_idx[i]
        img_root = os.path.join(self.ct_path,'image', tumormin_id)
        roi_root = os.path.join(self.ct_path,'roi', tumormin_id)
        img = sitk.ReadImage(img_root)
        roi = sitk.ReadImage(roi_root)

        img = sitk.GetArrayFromImage(img).astype(np.float32)
        roi = sitk.GetArrayFromImage(roi).astype(np.float32)
        img,roi = randomcrop(img,roi,self.size)
        img = torch.from_numpy(img)
        roi = torch.from_numpy(roi)
        text= self.text_embeddings[tumormin_id]['embeddings']
        if text.shape[0] > 1000:
            text = torch.mean(text[:1000,::], dim=0)
        else:
            text = torch.mean(text, dim=0)

        # text = self.all_label[tumormin_id]['image_findings'][:510]

        label = self.img_list[tumormin_id]
        
        return img,roi, text, label, tumormin_id #,os.path.basename(self.images_ind[i])

class BalancedBatchSampler(BatchSampler):
    def __init__(self, labels, n_classes, n_samplers):
        # super(BalancedBatchSampler, self).__init__()
        self.labels = np.array(labels)
        self.labels_set = list(set(self.labels))
        self.labels_to_indices = {
            label: np.where(self.labels == label)[0] for label in self.labels_set
        }
        for i in self.labels_set:
            np.random.shuffle(self.labels_to_indices[i])
 
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samplers
        self.batch_size = self.n_classes * self.n_samples
        self.n_dataset = len(self.labels)
 
    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                if self.used_label_indices_count[class_] + self.n_samples < len(self.labels_to_indices[class_]):
                    temp = self.labels_to_indices[class_][self.used_label_indices_count[class_]: self.used_label_indices_count[class_] + self.n_samples]
                    self.used_label_indices_count[class_] += self.n_samples
                else:
                    temp = self.labels_to_indices[class_][self.used_label_indices_count[class_]: len(self.labels_to_indices[class_])-1]
                    np.random.shuffle(self.labels_to_indices[class_])
                    temp =np.concatenate([temp,self.labels_to_indices[class_][:self.n_samples-len(temp)]],0)
                    self.used_label_indices_count[class_] = self.n_samples-len(temp)
                indices.extend(temp)
            # print(indices)
            yield indices
            self.count += self.n_classes * self.n_samples
 
    def __len__(self):
        return self.n_dataset // self.batch_size