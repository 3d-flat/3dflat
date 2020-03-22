import logging
import pathlib
import random
import shutil
import time
import os
from skimage.measure import compare_psnr

#os.environ["CUDA_VISIBLE_DEVICES"] ="1"
import sys

import matplotlib


from models.subsampling_model_3d import Subsampling_Model_3D

import numpy as np
import torch
import torchvision
from tensorboardX import SummaryWriter
from torch.nn import functional as F
from torch.utils.data import DataLoader
from common.args import Args
from data import transforms
from data.mri_data import DataTransform3D, SliceData3D
import pytorch_nufft.transforms
from evaluate import psnr
matplotlib.use('Agg')
import matplotlib.pyplot as plt
DEBUG=False
if not DEBUG:
    from mpl_toolkits.mplot3d import Axes3D

from scipy.spatial import distance_matrix
from common.utils import get_vel_acc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataTransform3D_Train(DataTransform3D):
    def __init__(self, resolution, depth,resolution_degrading):
        super().__init__(resolution, depth,resolution_degrading)

    def __call__(self, kspace, target, attrs, fname):
        kspace, target, mean, std=super().__call__(kspace, target, attrs, fname)
        return kspace, target, mean, std, 0



def create_datasets_3d(args):
    train_data = SliceData3D(
        root=args.data_path / 'brainT1/train',
        transform=DataTransform3D_Train(args.resolution,args.depth,args.resolution_degrading),
        sample_rate=args.sample_rate
    )
    dev_data = SliceData3D(
        root=args.data_path / 'brainT1/val',
        transform=DataTransform3D_Train(args.resolution, args.depth, args.resolution_degrading),
        sample_rate=args.sample_rate
    )
    return dev_data, train_data


def create_data_loaders_3d(args):
    dev_data, train_data = create_datasets_3d(args)
    display_data = [dev_data[i] for i in range(0, len(dev_data), len(dev_data) // 16)]

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    dev_loader = DataLoader(
        dataset=dev_data,
        batch_size=args.batch_size,
        num_workers=8,
        pin_memory=True,
    )
    display_loader = DataLoader(
        dataset=display_data,
        batch_size=1,
        num_workers=8,
        pin_memory=True,
    )
    return train_loader, dev_loader, display_loader


def train_epoch(args, epoch, model, data_loader, optimizer, writer):
    model.train()
    avg_loss = 0.

    if epoch>=args.weight_increase_epoch:
        args.vel_weight *= 1.5
        args.acc_weight *= 1.5
    start_epoch = start_iter = time.perf_counter()
    for iter, data in enumerate(data_loader):
        input, target, mean, std, norm = data
        input = input.unsqueeze(1).to(args.device)
        target = target.to(args.device)
        output = model(input).squeeze(1)
        x = model.get_trajectory()
        v, a = get_vel_acc(x)

        #During training, adjust the acc,vel for the coarse trajectory.
        acc_loss = args.acc_weight * torch.sqrt(torch.sum(
            torch.pow(F.softshrink(a, args.a_max * (args.realworld_points_per_shot / args.points_per_shot) ** 2 / 3),
                      2)))
        vel_loss = args.vel_weight * torch.sqrt(torch.sum(
            torch.pow(F.softshrink(v, args.v_max * args.realworld_points_per_shot / args.points_per_shot / 2), 2)))
        rec_loss = args.rec_weight * F.l1_loss(output, target)

        loss = rec_loss + vel_loss + acc_loss
        optimizer.zero_grad()
        loss.backward()
        if args.initialization == '2dstackofstars':
            x.grad[:,:,2]=0
        optimizer.step()
        avg_loss = 0.99 * avg_loss + 0.01 * loss.item() if iter > 0 else loss.item()
        if iter % args.report_interval == 0:
            logging.info(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item():.4g} Avg Loss = {avg_loss:.4g} '
                f'rec_loss: {rec_loss:.4g}, vel_loss: {vel_loss:.4g}, acc_loss: {acc_loss:.4g}, '
            )
        start_iter = time.perf_counter()
    return avg_loss, time.perf_counter() - start_epoch


def evaluate(args, epoch, model, data_loader, writer):
    model.eval()
    losses = []
    psnrs= []
    start = time.perf_counter()
    with torch.no_grad():
        if epoch != 0:
            for iter, data in enumerate(data_loader):
                input, target, mean, std, norm = data
                input = input.unsqueeze(1).to(args.device)
                target = target.to(args.device)
                output = model(input).squeeze(1)
                outputnorm,_,_=transforms.normalize_instance(output, eps=1e-11)
                psnrs.append( psnr(target.cpu().numpy(), outputnorm.cpu().numpy()))
                loss = args.rec_weight * F.l1_loss(output, target)
                losses.append(loss.item())

            x = model.get_trajectory()
            v, a = get_vel_acc(x)
            acc_loss = args.acc_weight * torch.sqrt(torch.sum(torch.pow(F.softshrink(a, args.a_max), 2)))
            vel_loss = args.vel_weight * torch.sqrt(torch.sum(torch.pow(F.softshrink(v, args.v_max), 2)))
            rec_loss =  np.mean(losses)
            psnr_avg = np.mean(psnrs)

            writer.add_scalar('Rec_Loss', rec_loss, epoch)
            writer.add_scalar('Acc_Loss', acc_loss.detach().cpu().numpy()/args.acc_weight, epoch)
            writer.add_scalar('Vel_Loss', vel_loss.detach().cpu().numpy()/args.vel_weight, epoch)
            writer.add_scalar('Acc_Weight',args.acc_weight,epoch)
            writer.add_scalar('Total_Loss',
                              rec_loss + acc_loss.detach().cpu().numpy() + vel_loss.detach().cpu().numpy(), epoch)
            writer.add_scalar('PSNR', psnr_avg, epoch)

        x = model.get_trajectory()
        v, a = get_vel_acc(x)
        writer.add_figure('Trajectory_Proj_XY', plot_trajectory(x.detach().cpu().numpy(),0,1), epoch)
        writer.add_figure('Trajectory_Proj_YZ', plot_trajectory(x.detach().cpu().numpy(),1,2), epoch)
        writer.add_figure('Trajectory_Proj_XZ', plot_trajectory(x.detach().cpu().numpy(),0,2), epoch)
        writer.add_figure('Trajectory_3D', plot_trajectory(x.detach().cpu().numpy(),d3=True), epoch)
        writer.add_figure('Accelerations_plot', plot_acc(a.cpu().numpy(), args.a_max), epoch)
        writer.add_figure('Velocity_plot', plot_acc(v.cpu().numpy(), args.v_max), epoch)
        writer.add_text('Coordinates', str(x.cpu().numpy()).replace(' ', ','), epoch)
    return np.mean(losses), time.perf_counter() - start


def plot_scatter(x):
    if not DEBUG:
        fig = plt.figure(figsize=[10, 10])
        ax = plt.axes(projection='3d')
        ax.axis([-45, 45, -45, 45])
        for i in range(x.shape[0]):
            ax.plot3D(x[i,:, 0], x[i,:, 1], x[i,:,2],'.')
    else:
        fig = plt.figure(figsize=[10, 10])
        ax = fig.add_subplot(1, 1, 1)
        ax.axis([-45, 45, -45, 45])
        for i in range(x.shape[0]):
            ax.plot(x[i, :, 0], x[i, :, 1], x[i, :, 2], '.')

    return fig


def plot_trajectory(x,dim1=0,dim2=1,d3=False):
    if d3==True and not DEBUG:
        fig = plt.figure(figsize=[10, 10])
        ax = plt.axes(projection='3d')
        ax.axis([-45, 45, -45, 45])
        for i in range(x.shape[0]):
            ax.plot3D(x[i,:, 0], x[i,:, 1], x[i,:,2])
    else:
        fig = plt.figure(figsize=[10, 10])
        ax = fig.add_subplot(1, 1, 1)
        ax.axis([-45, 45, -45, 45])
        for i in range(x.shape[0]):
            ax.plot(x[i,:,dim1],x[i,:,dim2])
    return fig


def plot_acc(a, a_max=None):
    fig, ax = plt.subplots(3, sharex=True)

    ax[0].plot(a[:,:, 0].T)
    ax[1].plot(a[:,:, 1].T)
    ax[2].plot(a[:,:, 2].T)
    if a_max != None:
        limit = np.ones(a.shape[1]) * a_max
        ax[2].plot(limit, color='red')
        ax[2].plot(-limit, color='red')
        ax[1].plot(limit, color='red')
        ax[1].plot(-limit, color='red')
        ax[0].plot(limit, color='red')
        ax[0].plot(-limit, color='red')
    return fig


def visualize(args, epoch, model, data_loader, writer):
    def save_image(image, tag):
        #iterate over all slices
        for i in range(image.shape[-3]//2-5,image.shape[-3]//2+5):
            im=image[:,:,i,:,:] #get slice
            im -= im.min()
            im /= im.max()
            grid = torchvision.utils.make_grid(im, nrow=4, pad_value=1)
            writer.add_image(tag+"_slice_{}".format(i), grid, epoch)

    model.eval()
    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            input, target, mean, std, norm = data
            input = input.unsqueeze(1).to(args.device)
            target = target.unsqueeze(1).to(args.device)
            if epoch==0:
                save_image(target, 'Target')
            if epoch != 0 or epoch==0:
                output = model(input.clone())
                corrupted = model.subsampling(input)

                save_image(output, 'Reconstruction')
                save_image(corrupted, 'Corrupted')
                save_image(torch.abs(target - output), 'Error')
            break


def save_model(args, exp_dir, epoch, model, optimizer, best_dev_loss, is_new_best):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_dev_loss': best_dev_loss,
            'exp_dir': exp_dir
        },
        f=exp_dir + '/model.pt'
    )
    if is_new_best:
        shutil.copyfile(exp_dir + '/model.pt', exp_dir + '/best_model.pt')


def build_model(args):
    model = Subsampling_Model_3D(
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
    return model


def load_model(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    args = checkpoint['args']
    model = build_model(args)
    if args.data_parallel:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint['model'])

    optimizer = build_optim(args, model)
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint, model, optimizer


def build_optim(args, model):
    optimizer = torch.optim.Adam([{'params': model.subsampling.parameters(), 'lr': args.sub_lr},
                                  {'params': model.reconstruction_model.parameters()}], args.lr)
    return optimizer


def train():
    args = create_arg_parser().parse_args()
    args.v_max = args.gamma * args.G_max * args.FOV * args.dt
    args.a_max = args.gamma * args.S_max * args.FOV * args.dt**2 * 1e3
    args.exp_dir = f'summary/{args.test_name}'
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    pathlib.Path(args.exp_dir).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=args.exp_dir)
    with open(args.exp_dir + '/args.txt', "w") as text_file:
        print(vars(args), file=text_file)

    if args.resume:
        checkpoint, model, optimizer = load_model(args.checkpoint)
        best_dev_loss = checkpoint['best_dev_loss']
        start_epoch = checkpoint['epoch'] + 1
        del checkpoint
    else:
        model = build_model(args)
        if args.data_parallel:
            model = torch.nn.DataParallel(model)
        optimizer = build_optim(args, model)
        best_dev_loss = 1e9
        start_epoch = 0
    logging.info(args)

    train_loader, dev_loader, display_loader = create_data_loaders_3d(args)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step_size, args.lr_gamma)
    dev_loss, dev_time = evaluate(args, 0, model, dev_loader, writer)
    visualize(args, 0, model, display_loader, writer)

    for epoch in range(start_epoch, args.num_epochs):
        train_loss, train_time = train_epoch(args, epoch, model, train_loader, optimizer, writer)
        dev_loss, dev_time = evaluate(args, epoch + 1, model, dev_loader, writer)
        visualize(args, epoch + 1, model, display_loader, writer)


        if dev_loss < best_dev_loss:
            is_new_best = True
            best_dev_loss = dev_loss
            best_epoch = epoch + 1
        else:
            is_new_best = False
        save_model(args, args.exp_dir, epoch, model, optimizer, best_dev_loss, is_new_best)
        logging.info(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
            f'DevLoss = {dev_loss:.4g} TrainTime = {train_time:.4f}s DevTime = {dev_time:.4f}s',
        )
    print(args.test_name)
    print(f'Training done, best epoch: {best_epoch}')
    writer.close()


def create_arg_parser():
    parser = Args()
    parser.add_argument('--test-name', type=str, default='test', help='name for the output dir')
    parser.add_argument('--exp-dir', type=pathlib.Path, default='summary/test',
                        help='Path where model and results should be saved')
    parser.add_argument('--resume', action='store_true',
                        help='If set, resume the training from a previous model checkpoint. '
                             '"--checkpoint" should be set with this')
    parser.add_argument('--checkpoint', type=str, default='summary/test/model.pt',
                        help='Path to an existing checkpoint. Used along with "--resume"')
    parser.add_argument('--report-interval', type=int, default=100, help='Period of loss reporting')


    parser.add_argument('--f_maps', type=int, default=32, help='Number of U-Net feature maps')
    parser.add_argument('--data-parallel', action='store_true', default=False,
                        help='If set, use multiple GPUs using data parallelism')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Which device to train on. Set to "cuda" to use the GPU')
    parser.add_argument('--acceleration-factor', default=20, type=int,
                        help='Number of shots in the multishot trajectory is calculated: res*depth/acceleration_factor')

    # optimization parameters
    parser.add_argument('--batch-size', default=16, type=int, help='Mini batch size')
    parser.add_argument('--num-epochs', type=int, default=40, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--lr-step-size', type=int, default=30,
                        help='Period of learning rate decay')
    parser.add_argument('--lr-gamma', type=float, default=0.01,
                        help='Multiplicative factor of learning rate decay')
    parser.add_argument('--weight-decay', type=float, default=0.,
                        help='Strength of weight decay regularization')
    parser.add_argument('--sub-lr', type=float, default=1e-1, help='lerning rate of the sub-samping layel')

    # trajectory learning parameters
    parser.add_argument('--trajectory-learning', default=True,action='store_false',
                        help='trajectory_learning, if set to False, fixed trajectory, only reconstruction learning.')

    parser.add_argument('--weight-increase-epoch', default=100, type=int, help='Epoch when the velocity and acceleration weights start increasing.')
    parser.add_argument('--acc-weight', type=float, default=1e-2, help='weight of the acceleration loss')
    parser.add_argument('--vel-weight', type=float, default=1e-1, help='weight of the velocity loss')
    parser.add_argument('--rec-weight', type=float, default=1, help='weight of the reconstruction loss')
    parser.add_argument('--gamma', type=float, default=42576, help='gyro magnetic ratio - kHz/T')
    parser.add_argument('--G-max', type=float, default=40, help='maximum gradient (peak current) - mT/m')
    parser.add_argument('--S-max', type=float, default=200, help='maximum slew-rate - T/m/s')
    parser.add_argument('--FOV', type=float, default=0.2, help='Field Of View - in m')
    parser.add_argument('--dt', type=float, default=1e-5, help='sampling time - sec')
    parser.add_argument('--realworld-points-per-shot', type=int, default=3000, help='Points sampled in every actual shot')
    parser.add_argument('--points-per-shot', type=int, default=500, help='Length of shot in learned trajectory - actual shot'
                                                                          'consists of missing points on the line')
    parser.add_argument('--a-max', type=float, default=0.17, help='maximum acceleration')
    parser.add_argument('--v-max', type=float, default=3.4, help='maximum velocity')
    parser.add_argument('--initialization', type=str, default='spiral',
                        help='Trajectory initialization ')
    return parser


if __name__ == '__main__':
    train()
