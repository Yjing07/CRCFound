from lifelines.utils import concordance_index
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"
from random import shuffle
import torch
import random
import logging
import utils as utils
import numpy as np
import json
from functools import partial
from lib.final_fusion import Fusion_survival
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report, confusion_matrix
import torch.nn.functional as F
import argparse
from utils import read_json
from sklearn import manifold
from lib.vit_fusion import ViT
import matplotlib.pyplot as plt
from utils.ct_dataset import MAE_fusion_class
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc, roc_auc_score, precision_recall_fscore_support,confusion_matrix


def bucketize(a: torch.Tensor, ids: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    mapping = {k.item(): v.item() for k, v in zip(a, ids)}

    # From `https://stackoverflow.com/questions/13572448`.
    palette, key = zip(*mapping.items())
    key = torch.tensor(key)
    palette = torch.tensor(palette)

    index = torch.bucketize(b.ravel(), palette)
    remapped = key[index].reshape(b.shape)

    return remapped

def precision_recall_f1(confusion_matrix):
    num_classes = confusion_matrix.shape[0]
    precision_list = []
    recall_list = []
    f1_list = []

    for i in range(num_classes):
        TP = confusion_matrix[i, i]
        FP = np.sum(confusion_matrix[:, i]) - TP
        FN = np.sum(confusion_matrix[i, :]) - TP

        precision = (TP+1e-15) / (TP + FP+1e-15)
        recall = (TP+1e-15) / (TP + FN+1e-15)

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(2 * (precision * recall) / (precision + recall))
    return precision_list,recall_list,f1_list

def save_confusion_matrix(num_classes,confusion_matrix,save_dir):
    # sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=list(range(num_classes)), yticklabels=list(range(num_classes)))
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig(save_dir) 
    plt.clf()
    return 

def Acc(confusion_matrix):
    return np.trace(confusion_matrix)/np.sum(confusion_matrix)

@torch.no_grad()
def return_auc(target_array,possibility_array,num_classes):
    enc = OneHotEncoder()
    target_onehot = enc.fit_transform(target_array.unsqueeze(1))
    target_onehot = target_onehot.toarray()
    class_auc_list = []
    for i in range(num_classes):
        class_i_auc = roc_auc_score(target_onehot[:,i], possibility_array[:,i])
        class_auc_list.append(class_i_auc)
    macro_auc = roc_auc_score(np.round(target_onehot,0), possibility_array, average="macro", multi_class="ovo")
    return macro_auc, class_auc_list

@torch.no_grad()
def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]

@torch.no_grad()
def five_scores(labels, predictions, num_classes, log_dir, all_patient_ids,logger,flag):
    
    this_class_label = list(np.argmax(predictions,axis=-1))
    val_dict = {}
    val_dict['label'] = [int(i) for i in labels]
    val_dict['pred'] = [i.tolist() for i in predictions]
    precision, recall, fscore, _ = precision_recall_fscore_support(labels, this_class_label, average='macro')
    acc=accuracy_score(labels, this_class_label)
    if num_classes>2:
        auc_value, class_auc_list = return_auc(torch.LongTensor(np.array(labels)), torch.Tensor(np.array(predictions)), num_classes)
    else:

        auc_value = roc_auc_score(labels, [i[1] for i in predictions])
        class_auc_list = []
    c_m = confusion_matrix(labels, this_class_label)
    return c_m, auc_value, class_auc_list, acc, precision, recall, fscore, val_dict

@torch.no_grad()
def evaluate(model, data_loader, args, logger,flag=0):
    model.eval()
    with torch.no_grad():
        pred_all = []
        label_all = []
        all_patient_id = []
        all_features = []
        for step, data in enumerate(data_loader):
            images,rois, text, label, patient_id = data
            for i in patient_id:
                all_patient_id.append(i)
            images = torch.unsqueeze(images,1).to(torch.float32).cuda()
            rois = torch.unsqueeze(rois,1).to(torch.float32).cuda()
            if args.model_name=='resnet18_fe':
                images = torch.cat([images,rois],1)
            else:
                images = images
            pred, _features = model(images, text, rois)
            all_features.append(_features)
            pred_all.extend(F.softmax(pred.detach(),dim=-1).cpu().numpy())
            label_all.extend(list(label.cpu().numpy()))

    c_m, auc_value, class_auc_list, accuracy, precision, recall, fscore, val_dict = five_scores(label_all, pred_all, args.num_classes, args.log_dir, all_patient_id,logger,flag)

    return c_m, auc_value, class_auc_list, accuracy, precision, recall, fscore, np.concatenate(all_features, axis=0), val_dict



def main(args, logger):
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    batch_size = args.batch_size

    ct_model = ViT(
    image_size = 256,          # image size
    frames = 32,               # number of frames
    image_patch_size = 16,     # image patch size
    frame_patch_size = 16,      # frame patch size
    num_classes=args.num_classes,
    dim = 1024,
    depth = 24,
    heads = 16,
    emb_dropout = 0.1,
    norm_layer=partial(torch.nn.LayerNorm, eps=1e-6))

    model = Fusion_survival(ct_model, args.num_classes)
    
    args.pretrained = args.pretrained + f'Fold{str(args.Fold)}-0.0001/models/the best auc model.pth'
    ct_weights_path = args.pretrained
    if os.path.exists(ct_weights_path):
        weights_dict = torch.load(ct_weights_path, map_location='cpu')
        model.load_state_dict(weights_dict)
        print('load sucessfully!!!')

    model = model.cuda()
  
    lab_dict = read_json(args.label_path)
    img_idx_list = list(lab_dict.keys())
    val_ind = img_idx_list['validation']

    valset = MAE_fusion_class(val_ind, args.data_path, lab_dict, args.shape)

    val_loader = torch.utils.data.DataLoader(valset,
                                             batch_size=batch_size,
                                             pin_memory=True,
                                             num_workers=opt.num_workers,
                                             shuffle=False,
                                             drop_last=False
                                             )
    
    print("Creating model")
    val_c_m, val_auc, val_auc_list, val_accuracy, val_precision, val_recall, val_fscore, val_features, val_dict = evaluate(model=model, data_loader=val_loader, args=args, logger=logger, flag=1)

    all_dict ={}
    all_dict['label'] = val_dict['label']
    all_dict['pred'] = val_dict['pred']

    target_dir = args.log_dir + '/' + 'pred.json'
    json.dump(all_dict, open(target_dir, 'w'))

    print("val auc: {}, val auc list: {}, val_acc: {}, val_precision: {}, val_recall: {}, val_fscore: {}".format(val_auc, val_auc_list, val_accuracy, val_precision, val_recall, val_fscore))
    print("val confusion matrix:{}".format(val_c_m))
    logger.info("val auc: {}, val auc list: {}, val_acc: {}, val_precision: {}, val_recall: {}, val_fscore: {}".format(val_auc, val_auc_list, val_accuracy, val_precision, val_recall, val_fscore))
    logger.info("val confusion matrix:{}".format(val_c_m))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='mae')
    parser.add_argument('--vit', default=True, type=bool)
    parser.add_argument('--pretrained', default='', type=str)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--scale', type=list, default=[1])
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--shape', type=tuple, default=(32,256,256))
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--lrf', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=0.4)
    parser.add_argument('--data_path', type=str, default="")
    parser.add_argument('--label_path', type=str, default="")
    parser.add_argument('--img_idx', type=str, default="")
    parser.add_argument('--log_path', type=str,
                        default='./logs',
                        help='path to log')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--ver', type=str,
                        default='/Fold1',
                        help='version of training')
    # 不要改该参数，系统会自动分
    parser.add_argument('--device', default='cuda:4', help='device id (i.e. 0 or 0,1 or cpu)')
   
    opt = parser.parse_args()
    
    exp_name = opt.model_name+ '-'+opt.ver
    opt.log_dir = opt.log_path+'/' + exp_name + '/logs'
    opt.model_dir = opt.log_path+'/' + exp_name + '/models'
    os.makedirs(opt.log_dir, exist_ok=True)
    os.makedirs(opt.model_dir, exist_ok=True)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]')
    console_handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(console_handler)
    file_handler = logging.FileHandler(os.path.join(opt.log_dir, 'train_log.log'))
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.info('Hyperparameter setting{}'.format(opt))

    main(opt, logger)