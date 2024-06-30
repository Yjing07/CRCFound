from lifelines.utils import concordance_index
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from random import shuffle
import torch
import random
import logging
import utils as utils
import numpy as np
import json
from functools import partial
import argparse
from utils.survival_dataset import KAD_Survival
from utils.utils import read_json
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
    True_pred = []
    event_all = []
    delay_all = []
    ids_all =[]
    for step, data in enumerate(data_loader):
        images,rois, event, delay, ids = data

        images = torch.unsqueeze(images,1).to(torch.float32).cuda()
        rois = torch.unsqueeze(rois,1).to(torch.float32).cuda()
        # images = torch.cat([images,rois],1)
        pred = model(images,rois)

        pred_all.extend(-pred)
        True_pred.extend(pred)
        event_all.extend(event.cuda())
        delay_all.extend(delay.cuda())
        ids_all.extend(ids)
        
    c_index = concordance_index(torch.tensor(delay_all),torch.tensor(pred_all),torch.tensor(event_all))
    c_index_torch = concordance_index_torch(torch.tensor(pred_all),torch.tensor(delay_all),torch.tensor(event_all))
    return c_index,c_index_torch, True_pred, event_all, delay_all,ids_all

def main(args):
    device = args.device
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    batch_size = args.batch_size

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
    if os.path.exists(ct_weights_path):
        weights_dict = torch.load(ct_weights_path, map_location='cpu')
        model.load_state_dict(weights_dict)
 
    model = model.cuda()

    lab_dict = read_json(args.label_path)
    img_idx_list = list(lab_dict.keys())
    val_ind = img_idx_list[args.Fold]['Training'] + img_idx_list[args.Fold]['Validation']
    valset = KAD_Survival(val_ind, args.data_path, lab_dict, args.shape, args.task_name)
    val_loader = torch.utils.data.DataLoader(valset,
                                             batch_size=batch_size,
                                             pin_memory=True,
                                             num_workers=opt.num_workers,
                                             shuffle=True,
                                             drop_last=False
                                             )
    
    print("Creating model")
    

    c_index,_, eval_pred_all, eval_event_all, eval_delay_all, val_ids = evaluate_c_index(model=model,    
                                            data_loader=val_loader,
                                            device=device, args=args)

    pred_all =  eval_pred_all
    event_all =  eval_event_all
    delay_all = eval_delay_all
    ids_all =  val_ids
    assert len(pred_all) == len(event_all) == len(delay_all)==len(ids_all)
    _all = {}
    _all['preds'] = [part.cpu().item() for part in pred_all]
    _all['event'] = [part.cpu().item() for part in event_all]
    _all['delay'] = [part.cpu().item() for part in delay_all]
    _all['ids'] = ids_all
    json.dump(_all, open(args.log_dir + f'/{args.Fold}_pred.json','w'))
    print(c_index)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str,
                        default='mae',
                        help='model name:[resnet18_fe,resnet34,resnet50_fe,resnet101,resnext50,resnext50_fe,resnext152_fe,resnet18_fe,resnet34_fe]')
    parser.add_argument('--vit', default=True, type=bool)
    parser.add_argument('--pretrained', default='', type=str)
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--scale', type=list, default=[1])
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--shape', type=tuple, default=(32,256,256))
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--task_name', type=str, default='dfs')
    parser.add_argument('--data_path', type=str, default="")
    parser.add_argument('--label_path', type=str, default="")
    parser.add_argument('--img_idx', type=str, default="")
    parser.add_argument('--log_path', type=str,
                        default='./logs',
                        help='path to log')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--ver', type=str,
                        default='/Fold2',
                        help='version of training')
    # 不要改该参数，系统会自动分
    parser.add_argument('--device', default='cuda:4', help='device id (i.e. 0 or 0,1 or cpu)')
   
    opt = parser.parse_args()
    if opt.pretrained:
        _m_name = 'pre'
    else:
        _m_name = 'sl'

    opt.log_path= opt.log_path + '/' + f'{opt.task_name}_{_m_name}'
    exp_name = opt.model_name+opt.ver+'-'+str(opt.lr)
    opt.log_dir = opt.log_path+'/' + exp_name + '/logs'
    os.makedirs(opt.log_dir, exist_ok=True)
    logging.info('Hyperparameter setting{}'.format(opt))

    print('Hyperparameter setting{}'.format(opt))
    main(opt)