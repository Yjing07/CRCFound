from lifelines.utils import concordance_index
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from random import shuffle
import torch
import random
import logging
import utils as utils
import numpy as np
import json
from collections import OrderedDict
from functools import partial
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import argparse
from utils.survival_dataset import KAD_Survival, BalancedBatchSampler
from utils import read_json
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import math 
from utils.loss import cox_loss_torch,nll_loss,CoxLoss,NegativeLogLikelihood,ClipLoss
from utils.metrics import concordance_index_torch
from lib.vit_ct import ViT


def bucketize(a: torch.Tensor, ids: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    mapping = {k.item(): v.item() for k, v in zip(a, ids)}

    # From `https://stackoverflow.com/questions/13572448`.
    palette, key = zip(*mapping.items())
    key = torch.tensor(key)
    palette = torch.tensor(palette)

    index = torch.bucketize(b.ravel(), palette)
    remapped = key[index].reshape(b.shape)

    return remapped

@torch.no_grad()
def evaluate_c_index(model, data_loader, device, args):
    model.eval()
    pred_all = []
    event_all = []
    delay_all = []
    ids_all =[]
    for step, data in enumerate(data_loader):
        images,rois, event, delay, ids = data

        images = torch.unsqueeze(images,1).to(torch.float32).cuda()
        pred = model(images)

        pred_all.extend(-pred)
        event_all.extend(event.cuda())
        delay_all.extend(delay.cuda())
        ids_all.extend(ids)
        
    c_index = concordance_index(torch.tensor(delay_all),torch.tensor(pred_all),torch.tensor(event_all))
    c_index_torch = concordance_index_torch(torch.tensor(pred_all),torch.tensor(delay_all),torch.tensor(event_all))
    return c_index,c_index_torch, pred_all, event_all, delay_all,ids_all

def train_one_epoch(model, optimizer, data_loader, device, epoch,writer,args):
    model.train()
    optimizer.zero_grad()
    pred_all = []
    event_all = []
    delay_all = []
    all_ids = []
    for step, data in enumerate(data_loader):
        images,rois, event, delay, ids = data
        images = torch.unsqueeze(images,1).to(torch.float32).cuda()
        pred = model(images)

        pred_all.extend(pred)
        event_all.extend(event.cuda())
        delay_all.extend(delay.cuda())
        all_ids.extend(ids)

        if args.loss == 'cox':
            loss_survival = cox_loss_torch(pred.squeeze(-1), delay.cuda(), event.cuda())
        elif args.loss == 'nll_loss':
            loss = nll_loss(pred, event.reshape(-1,1).to(device),delay.reshape(-1,1).to(device))
        elif args.loss == 'cox_loss':
            loss_survival = CoxLoss(delay.reshape(-1,1).to(device), event.reshape(-1,1).to(device),pred.squeeze(-1),device)
        elif args.loss == 'NegativeLogLikelihood_loss':
            NegativeLogLikelihood_loss = NegativeLogLikelihood(device)
            loss_survival = NegativeLogLikelihood_loss(pred, delay.reshape(-1,1).to(device), event.reshape(-1,1).to(device))+cox_loss_torch(pred.squeeze(-1), delay.to(device), event.to(device)) +CoxLoss(delay.reshape(-1,1).to(device), event.reshape(-1,1).to(device),pred.squeeze(-1),device)

        loss = loss_survival
        loss.backward()
        optimizer.step()

        writer.add_scalar('Loss/train', loss.item(), epoch * len(data_loader) + step)
        writer.add_scalar('Loss/loss_survival', loss_survival.item(), epoch * len(data_loader) + step)

        if step%20==0:
            print("[epoch {},step {}/{}] loss {}, loss_survival {} ".format(epoch,step,len(data_loader), round(loss.detach().cpu().item(), 4), round(loss_survival.detach().cpu().item(), 4)))
            logging.info("[epoch {},step {}/{}] loss {}, loss_survival {} ".format(epoch,step,len(data_loader), round(loss.detach().cpu().item(), 4), round(loss_survival.detach().cpu().item(), 4)))

    return pred_all,event_all, delay_all,all_ids

def main(args):
    device = args.device
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
        dim = 1024,
        depth = 8,
        heads = 8,
        emb_dropout = 0.1,
        decoder_embed_dim=512, 
        decoder_depth=4, 
        decoder_num_heads=8,
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
        print("load sucessful!!")

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
    
    trainset = KAD_Survival(train_ind, args.data_path, lab_dict, args.shape, args.task_name)
    valset = KAD_Survival(val_ind, args.data_path, lab_dict, args.shape, args.task_name)
    train_batch_sampler = BalancedBatchSampler(trainset.labels, n_classes=args.num_classes+1, n_samplers=args.batch_size//(args.num_classes+1))

    train_loader = torch.utils.data.DataLoader(trainset,
                                            batch_sampler=train_batch_sampler,
                                            shuffle=False)

    val_loader = torch.utils.data.DataLoader(valset,
                                             batch_size=batch_size,
                                             pin_memory=True,
                                             num_workers=opt.num_workers,
                                             shuffle=True,
                                             drop_last=True
                                             )
    
    print("Creating model")
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.02) #, momentum=0.9
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / (args.epochs))) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    val_best = 0
    for epoch in range(args.epochs):
        train_pred_all, train_event_all, train_delay_all, train_ids = train_one_epoch(model,
                        optimizer=optimizer,
                        data_loader=train_loader,
                        device=device,
                        epoch=epoch,
                        writer=summary_writer,
                        args=args)
        
        scheduler.step()
        c_index,c_index_torch, eval_pred_all, eval_event_all, eval_delay_all, val_ids = evaluate_c_index(model=model,    
                                                data_loader=val_loader,
                                                device=device, args=args)
        val_acc = float(c_index)
        if val_acc > val_best:
            val_best = val_acc
            torch.save(model.state_dict(), os.path.join(args.model_dir, f'the best model.pth'))      
            logging.info('***********************************')
            pred_all = train_pred_all + eval_pred_all
            event_all = train_event_all + eval_event_all
            delay_all = train_delay_all + eval_delay_all
            ids_all = train_ids +val_ids
            assert len(pred_all) == len(event_all) == len(delay_all)==len(ids_all)
            _all = {}
            _all['preds'] = [part.cpu().item() for part in pred_all]
            _all['event'] = [part.cpu().item() for part in event_all]
            _all['delay'] = [part.cpu().item() for part in delay_all]
            _all['ids'] = ids_all
            json.dump(_all, open(args.log_dir + f'/{args.Fold}_pred.json','w'))

        print("[epoch {}] val c_torch: {},val c: {}".format(epoch, round(float(c_index_torch), 4), round(float(c_index), 4)))
        logging.info("[epoch {}] val c_torch: {},val c: {}".format(epoch, round(float(c_index_torch), 4), round(float(c_index), 4)))
        summary_writer.add_scalar('val c', c_index, epoch)
    print('The best c_torch: {}'.format(val_best))
    logging.info('The best c_torch: {}'.format(val_best))
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='mae')
    parser.add_argument('--vit', default=True, type=bool)
    parser.add_argument('--pretrained', default='', type=str)
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--scale', type=list, default=[1])
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--shape', type=tuple, default=(32,256,256))
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--lrf', type=float, default=0.1)
    parser.add_argument('--loss', type=str, default="cox",help="[cox,nll_loss,cox_loss,NegativeLogLikelihood_loss]")
    parser.add_argument('--loss_ratio', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=0.4)
    parser.add_argument('--task_name', type=str, default='dfs')
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
    parser.add_argument('--device', default='cuda:4', help='device id (i.e. 0 or 0,1 or cpu)')
   
    opt = parser.parse_args()

    opt.log_path= opt.log_path + '/' + f'{opt.task_name}'
    exp_name = opt.model_name+opt.ver+'-'+str(opt.lr)
    opt.log_dir = opt.log_path+'/' + exp_name + '/logs'
    opt.model_dir = opt.log_path+'/' + exp_name + '/models'
    os.makedirs(opt.log_dir, exist_ok=True)
    os.makedirs(opt.model_dir, exist_ok=True)

    logging.basicConfig(filename=os.path.join(opt.log_dir, 'train_log.log'),
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info('Hyperparameter setting{}'.format(opt))

    print('Hyperparameter setting{}'.format(opt))
    main(opt)