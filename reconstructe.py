import pathlib
import sys
from collections import defaultdict
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
import sys
import scipy.io as sio
import numpy as np
import torch
from torch.utils.data import DataLoader

import pytorch_nufft
from common.args import Args
from common.utils import save_reconstructions
from data import transforms
from data.mri_data import SliceData3D, DataTransform3D
from models.subsampling_model_3d import Subsampling_Model_3D

class DataTransform3D_Rec(DataTransform3D):
    def __init__(self, resolution, depth,resolution_degrading):
        super().__init__(resolution, depth,resolution_degrading)

    def __call__(self, kspace, target, attrs, fname):

        kspace, target, mean, std = super().__call__(kspace, target, attrs, fname)
        return kspace, mean, std, fname


def create_data_loaders(args):
    data = SliceData3D(
        root=args.data_path / f'brainT1/{args.data_split}',
        transform=DataTransform3D_Rec(args.resolution,args.depth,args.resolution_degrading),
        sample_rate=1.
    )
    data_loader = DataLoader(
        dataset=data,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
    )
    return data_loader

def load_model(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    args = checkpoint['args']
    model =Subsampling_Model_3D(
        in_chans=1,
        out_chans=1,
        f_maps=args.f_maps,
        acceleration_factor=args.acceleration_factor,
        res=args.resolution,
        depth=args.depth,
        trajectory_learning=args.trajectory_learning,
        initialization=args.initialization,
        points_per_shot=args.points_per_shot
    ).to(args.device)
    if args.data_parallel:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint['model'])
    return model

def eval(args, model, data_loader):
    model.eval()
    reconstructions = defaultdict(list)
    with torch.no_grad():
        for (input, mean, std, fnames) in data_loader:
            input = input.unsqueeze(1).to(args.device)
            recons = model(input).to('cpu').squeeze(1)
            for i in range(recons.shape[0]):
                recons[i] = recons[i] * std[i] + mean[i]
                reconstructions[fnames[i]].append(recons[i].squeeze(0).numpy())

    return reconstructions

def corrupted_outputs(args, model, data_loader):
    model.eval()
    reconstructions = defaultdict(list)
    with torch.no_grad():
        for (input, mean, std, fnames) in data_loader:
            input = input.unsqueeze(1).to(args.device)
            recons = model.get_corrupted(input).to('cpu').squeeze(1)
            for i in range(recons.shape[0]):
                recons[i] = recons[i] * std[i] + mean[i]
                reconstructions[fnames[i]].append(recons[i].squeeze(0).numpy())

    return reconstructions

def reconstructe():
    args = create_arg_parser().parse_args(sys.argv[1:])
    args.checkpoint = f'summary/{args.test_name}/model.pt'
    args.out_dir = f'summary/{args.test_name}/rec'

    data_loader = create_data_loaders(args)
    model = load_model(args.checkpoint)
    model.eval()
    reconstructions = eval(args, model, data_loader)
    save_reconstructions(reconstructions, args.out_dir)
    corrupted = corrupted_outputs(args, model, data_loader)
    save_reconstructions(corrupted, f'summary/{args.test_name}/corrupted')


    x = model.get_trajectory()
    x = x.detach().cpu().numpy()
    sio.savemat(f'summary/{args.test_name}/traj.mat', {'x': x})


def create_arg_parser():
    parser = Args()
    parser.add_argument('--test-name', type=str, default='test', help='name for the output dir')
    parser.add_argument('--data-split', choices=['val', 'test'],default='val',
                        help='Which data partition to run on: "val" or "test"')
    parser.add_argument('--checkpoint', type=pathlib.Path,default='summary/test/checkpoint/best_model.pt',
                        help='Path to the U-Net model')
    parser.add_argument('--out-dir', type=pathlib.Path,default='summary/test/rec',
                        help='Path to save the reconstructions to')
    parser.add_argument('--batch-size', default=16, type=int, help='Mini-batch size')
    parser.add_argument('--num-shots', default=49, type=int, help='Mini-batch size')
    parser.add_argument('--device', type=str, default='cuda', help='Which device to run on')
    parser.add_argument('--resolution-degrading', type=int,
                        help='Size of Kernel which is used to degrade the resolution. '
                             'Will degrade with kernel and then crop down to resolution.')

    return parser


if __name__ == '__main__':
    reconstructe()
