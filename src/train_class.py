import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from random import shuffle
import torch
import random
import logging
import utils as utils
import numpy as np
import json
from functools import partial
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict
from sklearn.metrics import classification_report, confusion_matrix
import torch.nn.functional as F
import argparse
from utils import read_json
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import math 
from utils.loss import cox_loss_torch,nll_loss,CoxLoss,NegativeLogLikelihood,ClipLoss,FocalLoss
from utils.metrics import concordance_index_torch
from lib.vit_ct import ViT
import matplotlib.pyplot as plt
from utils.ct_dataset import MAE_class
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

def get_training_class_count(num_classes,label_dict,image_list):
    label_cal = {}
    for i in range(num_classes):
        label_cal[i] = []
    for key in image_list:
        if label_dict[key] in label_cal:
            label_cal[label_dict[key]].append(key)
    class_count = torch.zeros(len(label_cal.keys()))
    for key,values in label_cal.items():
        class_count[key] = len(values)
        # print(key,len(values))
    return class_count

def get_weight(args, lab_dict):
    image_list = lab_dict.keys()
    class_count = get_training_class_count(args.num_classes,lab_dict,image_list)
    weight = torch.tensor([sum(class_count)/i for i in class_count])
    weight = torch.nn.functional.softmax(torch.log(weight),dim=0)
    # weight = torch.nn.functional.softmax(weight,dim=0)
    return weight


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
def five_scores(labels, predictions, num_classes, all_patient_ids,flag):
    
    this_class_label = list(np.argmax(predictions,axis=-1))

    if flag:
        val_dict = {}
        for i in range (len(all_patient_ids)):
            _dic = {}
            name = all_patient_ids[i]
            _dic['label'] = int(labels[i])
            _dic['pred'] = int(this_class_label[i])
            val_dict[name] = _dic
    precision, recall, fscore, _ = precision_recall_fscore_support(labels, this_class_label, average='macro')
    acc=accuracy_score(labels, this_class_label)
    if num_classes>2:
        auc_value, class_auc_list = return_auc(torch.LongTensor(np.array(labels)), torch.Tensor(np.array(predictions)), num_classes)

    else:

        auc_value = roc_auc_score(labels, [i[1] for i in predictions])
        class_auc_list = []
    c_m = confusion_matrix(labels,  this_class_label)
    return c_m, auc_value, class_auc_list, acc, precision, recall, fscore

@torch.no_grad()
def evaluate(model, data_loader, args,flag=0):
    model.eval()
    with torch.no_grad():
        pred_all = []
        label_all = []
        all_patient_id = []

        for step, data in enumerate(data_loader):
            images,rois, label,patient_id  = data
            for i in patient_id:
                all_patient_id.append(i)
            images = torch.unsqueeze(images,1).to(torch.float32).cuda()
            if args.model_name=='resnet18_fe':
                images = torch.cat([images,rois],1)
                pred = model(images)
            else:
                images = images
                pred = model(images)

            pred_all.extend(F.softmax(pred.detach(),dim=-1).cpu().numpy())
            label_all.extend(list(label.cpu().numpy()))

    c_m, auc_value, class_auc_list, accuracy, precision, recall, fscore = five_scores(label_all, pred_all, args.num_classes,all_patient_id,flag)

    return c_m, auc_value, class_auc_list, accuracy, precision, recall, fscore, label_all, list(np.argmax(pred_all,axis=-1)), all_patient_id

def train_one_epoch(model, optimizer, data_loader, loss_fn, epoch, writer,args):
    
    model.train()

    for step, data in enumerate(data_loader):
        images,rois, label,_ = data
        images = torch.unsqueeze(images,1).to(torch.float32).cuda()

        label = label.cuda()
        images = images
        pred = model(images)
        loss = loss_fn(pred, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar('Loss/train', loss.item(), epoch * len(data_loader) + step)
        writer.add_scalar('Loss/loss_class', loss.item(), epoch * len(data_loader) + step)

        if step%10==0:
            print("[epoch {},step {}/ {}] loss {} ".format(epoch,step, len(data_loader), round(loss.detach().cpu().item(), 4)))
            logging.info("[epoch {},step {}/ {}] loss {} ".format(epoch,step, len(data_loader), round(loss.detach().cpu().item(), 4)))

    return 

def main(args):
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    batch_size = args.batch_size
    summary_writer = SummaryWriter(args.log_dir)

    model = ViT(
    image_size = 256,          # image size
    frames = 32,               # number of frames
    image_patch_size = 16,     # image patch size
    frame_patch_size = 16,      # frame patch size
    num_classes = args.num_classes,
    dim = 1024,
    depth = 24,
    heads = 16,
    emb_dropout = 0.1,
    mlp_ratio=4.,
    norm_layer=partial(torch.nn.LayerNorm, eps=1e-6)) 

    ct_weights_path = args.pretrained
    model_dict = model.state_dict()
    if os.path.exists(ct_weights_path):
        updata_dict = OrderedDict()
        weights_dict = torch.load(ct_weights_path, map_location='cpu')
        pretrained_dict = weights_dict['model']
        new_state_dict = OrderedDict()
        for k, v in pretrained_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        for k, v in new_state_dict.items():
            if k in model_dict:
                updata_dict[k]=v
        model_dict.update(updata_dict)
        model.load_state_dict(model_dict)

    model = model.cuda()
    for param in model.parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if 'blocks' not in name and 'mlp_head' in name:
            param.requires_grad = True
        elif 'blocks' in name and 'adapter' in name:
            param.requires_grad = True
        elif 'prompt' in name:
            param.requires_grad = True
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

    lab_dict = read_json(args.label_path)
    img_idx_list = list(lab_dict.keys())
    train_ind = img_idx_list[args.Fold]['training']
    val_ind = img_idx_list[args.Fold]['validation']
    
    trainset = MAE_class(train_ind, args.data_path, lab_dict, args.shape)
    valset = MAE_class(val_ind, args.data_path, lab_dict, args.shape)

    train_loader = torch.utils.data.DataLoader(trainset,
                                            batch_size=batch_size,
                                             pin_memory=True,
                                             num_workers=opt.num_workers,
                                             shuffle=True,
                                             drop_last=True)

    val_loader = torch.utils.data.DataLoader(valset,
                                             batch_size=batch_size,
                                             pin_memory=True,
                                             num_workers=opt.num_workers,
                                             shuffle=True,
                                             drop_last=True
                                             )
    
    print("Creating model")
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.02) #, momentum=0.9
    lf = lambda x: ((1 + math.cos(x * math.pi / (args.epochs))) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    loss_fn = torch.nn.CrossEntropyLoss(weight=get_weight(args, train_ind)).cuda()
    val_best = 0
    for epoch in range(args.epochs):
        train_one_epoch(model,
                        optimizer=optimizer,
                        data_loader=train_loader,
                        loss_fn =loss_fn,
                        epoch=epoch,
                        writer=summary_writer,
                        args=args)
        
        scheduler.step()

        val_c_m, val_auc, val_auc_list, val_accuracy, val_precision, val_recall, val_fscore, labels_val, pres_val, ids_val = evaluate(model=model, data_loader=val_loader, args=args,flag=1)
        if val_auc > val_best:
            _all = {}
            all_labels = labels_val
            all_preds =  pres_val
            all_ids =  ids_val
            assert len(all_labels) == len(all_preds) == len(all_ids)
            _all['ids'] = all_ids
            _all['preds'] = [int(part) for part in all_preds]
            _all['labels'] = [int(part) for part in all_labels]
            json.dump(_all, open(args.log_dir + f'/{args.Fold}_pred.json','w'))
            val_best = val_auc
            torch.save(model.state_dict(), os.path.join(args.model_dir, f'the best auc model.pth'))      
            logging.info('***********************************')

        print("[epoch {}] val auc: {}, val auc list: {}, val_acc: {}, val_precision: {}, val_recall: {}, val_fscore: {}".format(epoch, val_auc, val_auc_list, val_accuracy, val_precision, val_recall, val_fscore))
        print("[epoch {}] val confusion matrix:{}".format(epoch, val_c_m))
        logging.info("[epoch {}] val auc: {}, val auc list: {}, val_acc: {}, val_precision: {}, val_recall: {}, val_fscore: {}".format(epoch, val_auc, val_auc_list, val_accuracy, val_precision, val_recall, val_fscore))
        logging.info("[epoch {}] val confusion matrix:{}".format(epoch, val_c_m))

        summary_writer.add_scalar('val auc', val_auc, epoch)
        summary_writer.add_scalar('val acc', val_accuracy, epoch)
        summary_writer.add_scalar('val prec', val_precision, epoch)
        summary_writer.add_scalar('val rec', val_recall, epoch)
        summary_writer.add_scalar('val f1', val_fscore, epoch)

    print('The best val auc: {}'.format(val_best))
    logging.info('The best val auc: {}'.format(val_best))
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='mae')
    parser.add_argument('--vit', default=True, type=bool)
    parser.add_argument('--checkpoint', default='', type=str)
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--scale', type=list, default=[1])
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--shape', type=tuple, default=(32,256,256))
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--lrf', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=0.4)
    parser.add_argument('--data_path', type=str, default="")
    parser.add_argument('--label_path', type=str, default="")
    parser.add_argument('--log_path', type=str,
                        default='./log',
                        help='path to log')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--ver', type=str,
                        default='/Fold0',
                        help='version of training')
    parser.add_argument('--device', default='cuda:4', help='device id (i.e. 0 or 0,1 or cpu)')
   
    opt = parser.parse_args()
    
    exp_name = opt.model_name+opt.ver+'-'+str(opt.lr)
    opt.log_dir = opt.log_path+'/' + exp_name + '/logs'
    opt.model_dir = opt.log_path+'/' + exp_name + '/models'
    os.makedirs(opt.log_dir, exist_ok=True)
    os.makedirs(opt.model_dir, exist_ok=True)

    logging.basicConfig(filename=os.path.join(opt.log_dir, 'train_log.log'),
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')

    logging.info('Hyperparameter setting{}'.format(opt))
    main(opt)