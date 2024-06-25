import os
import pathlib
import pprint

import SimpleITK as sitk
import numpy as np
import pandas as pd
import torch
import yaml
from matplotlib import pyplot as plt
from monai.data import decollate_batch
from inference_util import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    EnsureType
)
from numpy import logical_and as l_and, logical_not as l_not
from torch import distributed as dist
import nibabel as nib
import torch.nn as nn
import json
import math

import torch.distributed as dist

import json
import os
import SimpleITK as sitk
import numpy as np
import torch
import torch.distributed as dist
import random
import time,datetime
from collections import defaultdict, deque
random.seed(42)

def write_to_json(list1,save_root):
    if not os.path.exists(os.path.dirname(save_root)):
        os.makedirs(os.path.dirname(save_root))
    with open(save_root,'w') as f:
        json.dump(list1,f)

def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"]) #表示进程的序号，每个进程对应于一个rank
        args.world_size = int(os.environ['WORLD_SIZE']) #全局的并行数，rank的数量
        args.gpu = int(os.environ['LOCAL_RANK']) #一台机器上进程的相对序号
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID']) #可用作全局rank
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'  # 通信后端，nvidia GPU推荐使用NCCL
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)
    dist.barrier()

def read_json(file_root):
    with open(file_root, 'r') as f:
        js1 = json.load(f)
    return js1

def cleanup():
    dist.destroy_process_group()

def reduce_value(value, average=True):
    world_size = dist.get_world_size()
    if world_size < 2:  # 单GPU的情况
        return value

    with torch.no_grad():
        dist.all_reduce(value)
        if average:
            value /= world_size

        return value

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

def crop(img,roi,size):
    c,h,w =img.shape
    cc,hh,ww = size
    ci=0#random.randint(0,c-cc)
    hi=(h-hh)//2#random.randint(0,h-hh)
    wi=(w-ww)//2#random.randint(0,w-ww)
    # return img[ci:ci+cc],roi[ci:ci+cc]
    return img[ci:ci+cc,hi:hi+hh,wi:wi+ww],roi[ci:ci+cc,hi:hi+hh,wi:wi+ww]

def resize_image_itk(itkimage, newSize, resamplemethod=sitk.sitkNearestNeighbor):

    resampler = sitk.ResampleImageFilter()
    originSize = itkimage.GetSize()  # 原来的体素块尺寸
    originSpacing = itkimage.GetSpacing()
    newSize = np.array(newSize,float)
    factor = originSize / newSize
    newSpacing = originSpacing * factor
    newSize = newSize.astype(np.int) #spacing肯定不能是整数
    # print(newSize)
    resampler.SetReferenceImage(itkimage)  # 需要重新采样的目标图像
    resampler.SetSize(newSize.tolist())
    resampler.SetOutputSpacing(newSpacing.tolist())
    # resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itkimgResampled = resampler.Execute(itkimage)  # 得到重新采样后的图像
    return itkimgResampled

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def global_avg(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {:.4f}".format(name, meter.global_avg)
            )
        return self.delimiter.join(loss_str)    
    
    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))
        


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def compute_acc(logits, label, reduction='mean'):
    ret = (torch.argmax(logits, dim=1) == label).float()
    if reduction == 'none':
        return ret.detach()
    elif reduction == 'mean':
        return ret.mean().item()

def compute_n_params(model, return_str=True):
    tot = 0
    for p in model.parameters():
        w = 1
        for x in p.shape:
            w *= x
        tot += w
    if return_str:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)
    
def all_reduce_tensor(tensor, op=dist.ReduceOp.SUM, world_size=1):
    tensor = tensor.clone()
    dist.all_reduce(tensor, op)
    tensor.div_(world_size)
    return tensor

def save_args(args):
    """Save parsed arguments to config file.
    """
    config = vars(args).copy()
    del config['save_folder']
    del config['seg_folder']
    config_file = args.save_folder / (args.exp_name + ".yaml")
    with open(config_file, "w") as file:
        yaml.dump(config, file)

def master_do(func, *args, **kwargs):
    """Help calling function only on the rank0 process id ddp"""
    try:
        rank = dist.get_rank()
        if rank == 0:
            return func(*args, **kwargs)
    except AssertionError:
        # not in DDP setting, just do as usual
        func(*args, **kwargs)


def save_checkpoint(state: dict, save_folder: pathlib.Path):
    """Save Training state."""
    best_filename = str(save_folder) + 'model_best' + "_" + str(state["epoch"]) + '.pth.tar'
    torch.save(state, best_filename)


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    @staticmethod
    def _get_batch_fmtstr(num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def reload_ckpt(args, model, device=torch.device("cuda:0")):
    if os.path.isfile(args):
        print("=> loading checkpoint '{}'".format(args))
        checkpoint = torch.load(args, map_location=device)
        model = nn.DataParallel(model)
        model.load_state_dict(checkpoint['state_dict'], strict = True)
        model = model.module
    else:
        raise ValueError("=> no checkpoint found at '{}'".format(args))

def reload_ckpt_bis_unet(ckpt, model, temp_model, device=torch.device("cuda:0")):
    if os.path.isfile(ckpt):
        print(f"=> loading checkpoint {ckpt}")
        
        
        checkpoint = torch.load(ckpt, map_location=device)
        start_epoch = checkpoint['epoch']
        temp_model = nn.DataParallel(temp_model)
        temp_model.load_state_dict(checkpoint['state_dict'])
        temp_model = temp_model.module
        
        model.swin_unet.layers.load_state_dict(temp_model.swin_unet.layers.state_dict())
        model.swin_unet.layers_up.load_state_dict(temp_model.swin_unet.layers_up.state_dict())
        model.swin_unet.concat_back_dim.load_state_dict(temp_model.swin_unet.concat_back_dim.state_dict())
        model.swin_unet.limage = nn.Parameter(temp_model.swin_unet.limage.clone())
        return start_epoch
    # except RuntimeError:
        # TO account for checkpoint from Alex nets
        print("Loading model Alex style")
        model.load_state_dict(torch.load(ckpt, map_location='cpu'))
    else:
        raise ValueError(f"=> no checkpoint found at '{ckpt}'")
        
def reload_ckpt_bis(ckpt, model, temp_model, device=torch.device("cuda:0")):
    if os.path.isfile(ckpt):
        print(f"=> loading checkpoint {ckpt}")
        
        
        checkpoint = torch.load(ckpt, map_location=device)
        start_epoch = checkpoint['epoch']
        temp_model = nn.DataParallel(temp_model)
        temp_model.load_state_dict(checkpoint['state_dict'])
        temp_model = temp_model.module
        
        model.swin_unet.layers.load_state_dict(temp_model.swin_unet.layers.state_dict())
        model.swin_unet.layers_up.load_state_dict(temp_model.swin_unet.layers_up.state_dict())
        model.swin_unet.concat_back_dim.load_state_dict(temp_model.swin_unet.concat_back_dim.state_dict())
        model.swin_unet.limage = nn.Parameter(temp_model.swin_unet.limage.clone())
        return start_epoch
    # except RuntimeError:
        # TO account for checkpoint from Alex nets
        print("Loading model Alex style")
        model.load_state_dict(torch.load(ckpt, map_location='cpu'))
    else:
        raise ValueError(f"=> no checkpoint found at '{ckpt}'")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_metrics(preds, targets, patient, tta=False):
    """

    Parameters
    ----------
    preds:
        torch tensor of size 1*C*Z*Y*X
    targets:
        torch tensor of same shape
    patient :
        The patient ID
    tta:
        is tta performed for this run
    """
    pp = pprint.PrettyPrinter(indent=4)
    assert preds.shape == targets.shape, "Preds and targets do not have the same size"

    labels = ["ET", "TC", "WT"]

    metrics_list = []

    for i, label in enumerate(labels):
        metrics = dict(
            patient_id=patient,
            label=label,
            tta=tta,
        )

        if np.sum(targets[i]) == 0:
            print(f"{label} not present for {patient}")
            dice = 1 if np.sum(preds[i]) == 0 else 0

        else:
            tp = np.sum(l_and(preds[i], targets[i]))
            fp = np.sum(l_and(preds[i], l_not(targets[i])))
            fn = np.sum(l_and(l_not(preds[i]), targets[i]))

            dice = 2 * tp / (2 * tp + fp + fn)

        metrics[DICE] = dice
        pp.pprint(metrics)
        metrics_list.append(metrics)

    return metrics_list


def save_metrics(epoch, metrics, writer, current_epoch, teacher=False, save_folder=None):
    metrics = list(zip(*metrics))
    # print(metrics)
    # TODO check if doing it directly to numpy work
    metrics = [torch.tensor(dice, device="cpu").numpy() for dice in metrics]
    # print(metrics)
    labels = ("ET", "TC", "WT")
    metrics = {key: value for key, value in zip(labels, metrics)}
    # print(metrics)
    '''
    fig, ax = plt.subplots()
    ax.set_title("Dice metrics")
    ax.boxplot(metrics.values(), labels=metrics.keys())
    ax.set_ylim(0, 1)
    writer.add_figure(f"val/plot", fig, global_step=epoch)
   '''
    print(f"Epoch {current_epoch} :{'val' + '_teacher :' if teacher else 'Val :'}",
          [f"{key} : {np.nanmean(value)}" for key, value in metrics.items()])
    with open(f"{save_folder}/val{'_teacher' if teacher else ''}.txt", mode="a") as f:
        print(f"Epoch {current_epoch} :{'val' + '_teacher :' if teacher else 'Val :'}",
              [f"{key} : {np.nanmean(value)}" for key, value in metrics.items()], file=f)
    for key, value in metrics.items():
        tag = f"val{'_teacher' if teacher else ''}{''}/{key}_Dice"
        writer.add_scalar(tag, np.nanmean(value), global_step=epoch)
        


dice_metric = DiceMetric(include_background=True, reduction="mean")
dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")

post_trans = Compose(
    [EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold_values=True)]
)

VAL_AMP = True


# define inference method
def inference(input, model, patch_shape = 128):
    def _compute(input):
        return sliding_window_inference(
            inputs=input,
            roi_size=(patch_shape, patch_shape, patch_shape),
            sw_batch_size=1,
            predictor=model,
            overlap=0.5
        )

    if VAL_AMP:
        with torch.cuda.amp.autocast():
            return _compute(input)
    else:
        return _compute(input)

        
def generate_segmentations_monai_withlabel(data_loader, model, writer_1, args):
    metrics_list = []
    model.eval()
    for idx, val_data in enumerate(data_loader):
        torch.cuda.empty_cache()
        print(f"Validating case {idx}")
        patient_id = val_data["patient_id"][0]
        ref_path = val_data["seg_path"][0]
        crops_idx = val_data["crop_indexes"]

        ref_seg_img = sitk.ReadImage(ref_path)
        ref_seg = sitk.GetArrayFromImage(ref_seg_img)

        val_inputs, val_labels = (
            val_data["image"].cuda(),
            val_data["label"],
        )

        with torch.no_grad():
            val_outputs_1 = inference(val_inputs, model)
            val_outputs = [post_trans(i) for i in decollate_batch(val_outputs_1)]
        torch.cuda.empty_cache()
        segs = torch.zeros((1, 3, ref_seg.shape[0], ref_seg.shape[1], ref_seg.shape[2]))
        segs[0, :, slice(*crops_idx[0]), slice(*crops_idx[1]), slice(*crops_idx[2])] = val_outputs[0]
        #print(segs.max(), segs.min())
        segs = segs[0].numpy() > 0.5
        #print(segs.shape, segs.sum())
        
        '''
        et = segs[0]
        net = np.logical_and(segs[1], np.logical_not(et))
        ed = np.logical_and(segs[2], np.logical_not(segs[1]))
        
        labelmap = np.zeros((155, 240, 240), dtype=np.uint8)
        labelmap[et] = 4
        labelmap[net] = 1
        labelmap[ed] = 2
        
        labelmap = labelmap.transpose((2,1,0))
        
        labelmap = np.concatenate([labelmap, np.zeros((240,240,5), dtype=np.uint8)], 2)
        
        print(labelmap.shape)
        
        nib.save(nib.Nifti1Image(labelmap, None), f"{args.seg_folder_1}/{patient_id}.nii.gz")
        continue
        '''
        #labelmap = sitk.GetImageFromArray(labelmap)

        refmap_et, refmap_tc, refmap_wt = [np.zeros_like(ref_seg) for i in range(3)]
        refmap_et = ref_seg == 4
        refmap_tc = np.logical_or(refmap_et, ref_seg == 1)
        refmap_wt = np.logical_or(refmap_tc, ref_seg == 2)
        refmap = np.stack([refmap_et, refmap_tc, refmap_wt])

        patient_metric_list = calculate_metrics(segs, refmap, patient_id)
        metrics_list.append(patient_metric_list)
        #labelmap.CopyInformation(ref_seg_img)

        #print(f"Writing {args.seg_folder_1}/{patient_id}.nii.gz")
        #sitk.WriteImage(labelmap, f"{args.seg_folder_1}/{patient_id}.nii.gz")
    
    val_metrics = [item for sublist in metrics_list for item in sublist]
    df = pd.DataFrame(val_metrics)
    # overlap = df.boxplot(METRICS[1:2], by="label", return_type="axes")
    # overlap_figure = overlap[0].get_figure()
    # writer_1.add_figure("benchmark/overlap_measures", overlap_figure)
    # haussdorf_figure = df.boxplot(METRICS[0], by="label").get_figure()
    # writer_1.add_figure("benchmark/distance_measure", haussdorf_figure)
    # grouped_df = df.groupby("label")[METRICS]
    # summary = grouped_df.mean().to_dict()
    # for metric, label_values in summary.items():
        # for label, score in label_values.items():
            # writer_1.add_scalar(f"benchmark_{metric}/{label}", score)
    df.to_csv((args.save_folder_1 / 'results.csv'), index=False)


def generate_segmentations_monai(data_loader, model, writer_1, args):
    metrics_list = []
    model.eval()
    for idx, val_data in enumerate(data_loader):
        torch.cuda.empty_cache()
        print(f"Validating case {idx}")
        patient_id = val_data["patient_id"][0]
        ref_path = val_data["seg_path"][0]
        crops_idx = val_data["crop_indexes"]

        ref_seg_img = sitk.ReadImage(ref_path)
        ref_seg = sitk.GetArrayFromImage(ref_seg_img)

        val_inputs, val_labels = (
            val_data["image"].cuda(),
            val_data["label"],
        )

        with torch.no_grad():
            val_outputs_1 = inference(val_inputs, model)
            val_outputs = [post_trans(i) for i in decollate_batch(val_outputs_1)]
        torch.cuda.empty_cache()
        segs = torch.zeros((1, 3, ref_seg.shape[0], ref_seg.shape[1], ref_seg.shape[2]))
        segs[0, :, slice(*crops_idx[0]), slice(*crops_idx[1]), slice(*crops_idx[2])] = val_outputs[0]
        print(segs.max(), segs.min())
        segs = segs[0].numpy() > 0.5
        print(segs.shape, segs.sum())
        if segs[0].sum() < 500:
            
            et = np.zeros_like(segs[0])
        else:
            et = segs[0]
        net = np.logical_and(segs[1], np.logical_not(et))
        ed = np.logical_and(segs[2], np.logical_not(segs[1]))
        labelmap = np.zeros((155, 240, 240), dtype=np.uint8)
        labelmap[et] = 4
        labelmap[net] = 1
        labelmap[ed] = 2
        
        labelmap = labelmap.transpose((2,1,0))
        
        labelmap = np.concatenate([labelmap, np.zeros((240,240,5), dtype=np.uint8)], 2)
        
        print(labelmap.shape)
        
        nib.save(nib.Nifti1Image(labelmap, None), f"{args.seg_folder_1}/{patient_id}.nii.gz")
        continue
        labelmap = sitk.GetImageFromArray(labelmap)

        refmap_et, refmap_tc, refmap_wt = [np.zeros_like(ref_seg) for i in range(3)]
        refmap_et = ref_seg == 4
        refmap_tc = np.logical_or(refmap_et, ref_seg == 1)
        refmap_wt = np.logical_or(refmap_tc, ref_seg == 2)
        refmap = np.stack([refmap_et, refmap_tc, refmap_wt])

        patient_metric_list = calculate_metrics(segs, refmap, patient_id)
        metrics_list.append(patient_metric_list)
        labelmap.CopyInformation(ref_seg_img)

        print(f"Writing {args.seg_folder_1}/{patient_id}.nii.gz")
        sitk.WriteImage(labelmap, f"{args.seg_folder_1}/{patient_id}.nii.gz")
    
    exit(0)
    val_metrics = [item for sublist in metrics_list for item in sublist]
    df = pd.DataFrame(val_metrics)
    overlap = df.boxplot(METRICS[1:], by="label", return_type="axes")
    overlap_figure = overlap[0].get_figure()
    writer_1.add_figure("benchmark/overlap_measures", overlap_figure)
    haussdorf_figure = df.boxplot(METRICS[0], by="label").get_figure()
    writer_1.add_figure("benchmark/distance_measure", haussdorf_figure)
    grouped_df = df.groupby("label")[METRICS]
    summary = grouped_df.mean().to_dict()
    for metric, label_values in summary.items():
        for label, score in label_values.items():
            writer_1.add_scalar(f"benchmark_{metric}/{label}", score)
    df.to_csv((args.save_folder_1 / 'results.csv'), index=False)


HAUSSDORF = "haussdorf"
DICE = "dice"
SENS = "sens"
SPEC = "spec"
SSIM = "ssim"
METRICS = [HAUSSDORF, DICE, SENS, SPEC, SSIM]

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr