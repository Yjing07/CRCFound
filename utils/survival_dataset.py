import os
import SimpleITK as sitk
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from utils.utils import resize_image_itk,randomcrop,crop,read_json
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

class KAD_Survival(Dataset):

    def __init__(self, img_ind, ct_data_path, lab_dict, size, task_name):
        super(KAD_Survival,self).__init__()

        self.images_idx = img_ind
        self.ct_path = ct_data_path
        self.patient_slide = {}
        self.all_label = lab_dict
        self.size = size
        self.task_name = task_name
        self.labels = np.array([self.all_label[i][f'{task_name}.event'] for i in self.images_idx])

    def __len__(self):
        return len(self.images_idx)
    
    def __getitem__(self, i):
        # patient_id = self.images_idx[i]
        # tumormin_id = self.all_label[patient_id]['tuominID']
        tumormin_id = self.images_idx[i]
        img_root = os.path.join(self.ct_path,'image', tumormin_id)
        roi_root = os.path.join(self.ct_path,'roi', tumormin_id)
        img = sitk.ReadImage(img_root)
        roi = sitk.ReadImage(roi_root)

        img = sitk.GetArrayFromImage(img).astype(np.float32)
        roi = sitk.GetArrayFromImage(roi).astype(np.float32)
        # img,roi = randomcrop(img,roi,self.size)
        img = torch.from_numpy(img)
        roi = torch.from_numpy(roi)
        
        event =  torch.tensor(self.all_label[tumormin_id][f'{self.task_name}.event'])
        delay =  torch.tensor(self.all_label[tumormin_id][f'{self.task_name}.delay'])
 
        return img,roi, event, delay, tumormin_id

class Survival_fusion(Dataset):
    """自定义数据集"""

    def __init__(self, img_ind, ct_data_path, lab_dict, size, task_name):
        super(Survival_fusion,self).__init__()

        self.images_idx = img_ind
        self.ct_path = ct_data_path
        self.patient_slide = {}
        self.all_label = lab_dict
        self.size = size
        self.task_name = task_name
        self.labels = np.array([self.all_label[i][f'{task_name}.event'] for i in self.images_idx])
        self.text_embeddings = joblib.load('paitents_caption_embeddings1072.pkl')

    def __len__(self):
        return len(self.images_idx)
    
    def __getitem__(self, i):
        # patient_id = self.images_idx[i]
        # tumormin_id = self.all_label[patient_id]['tuominID']
        tumormin_id = self.images_idx[i]
        img_root = os.path.join(self.ct_path,'image', tumormin_id)
        roi_root = os.path.join(self.ct_path,'roi', tumormin_id)
        img = sitk.ReadImage(img_root)
        roi = sitk.ReadImage(roi_root)

        img = sitk.GetArrayFromImage(img).astype(np.float32)
        roi = sitk.GetArrayFromImage(roi).astype(np.float32)
        # img,roi = randomcrop(img,roi,self.size)
        img = torch.from_numpy(img)
        roi = torch.from_numpy(roi)
        
        event =  torch.tensor(self.all_label[tumormin_id][f'{self.task_name}.event'])
        delay =  torch.tensor(self.all_label[tumormin_id][f'{self.task_name}.delay'])
        text= self.text_embeddings[tumormin_id]['embeddings']
        if text.shape[0] > 1000:
            text = torch.mean(text[:500,::], dim=0)
        else:
            text = torch.mean(text, dim=0)

        # ct_caption = self.all_label[tumormin_id]['image_findings'][:510]
 
        return img,roi, event, delay, text, tumormin_id  #,os.path.basename(self.images_ind[i])

class BalancedBatchSampler(BatchSampler):
    def __init__(self, labels, n_classes, n_samplers):
        # super(BalancedBatchSampler, self).__init__()
        self.labels = labels
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