import torch
from scipy.interpolate import interp1d
from torch import nn
from torch.nn import functional as F

from common import args
from common.utils import get_vel_acc
from models.rec_models.unet_model import UnetModel
from models.rec_models.unet_model import UNet3D
import pytorch_nufft.nufft as nufft
import pytorch_nufft.interp as interp
import data.transforms as transforms
from scipy.spatial import distance_matrix
import numpy as np
import scipy.io as sio
from torchkbnufft import AdjKbNufft,KbNufft
class Subsampling_Layer_3D(nn.Module):
    def initilaize_trajectory(self,trajectory_learning,initialization,points_per_shot):

        num_measurements_shot=points_per_shot
        num_shots=self.res*self.depth//self.acceleration_factor

        if initialization=='sticks':
            num_shots=int(np.sqrt(num_shots))
            x = np.zeros((num_shots**2, num_measurements_shot, 3))
            r = np.linspace(-1, 1, num_measurements_shot)
            theta = np.linspace(0, np.pi, num_shots)
            phi = np.linspace(0, np.pi, num_shots)
            i=0
            for lgn in theta:
                for lat in phi:
                    x[i, :, 0] = r * np.sin(lgn) * np.cos(lat) * self.res / 2
                    x[i, :, 1] = r * np.sin(lgn) * np.sin(lat) * self.res / 2
                    x[i, :, 2] = r * np.cos(lgn) * self.depth / 2
                    i=i+1
            x = torch.from_numpy(x).float()
        elif initialization=='radial':
            # based on matlab spiral
            x = np.zeros((num_shots, num_measurements_shot, 3))
            r = np.linspace(-1, 1, num_measurements_shot)
            if num_shots==312:
                theta = np.load(f'spiral/theta312.npy')[0]
                phi = np.load(f'spiral/phi312.npy')[0]
            elif num_shots==320:
                load = sio.loadmat(f'spiral/320.mat')
                theta = load['theta'][0]
                phi = load['phi'][0]
            elif num_shots==640:
                load = sio.loadmat(f'spiral/640.mat')
                theta = load['theta'][0]
                phi = load['phi'][0]
            elif num_shots==64:
                load = sio.loadmat(f'spiral/64.mat')
                theta = load['theta'][0]
                phi = load['phi'][0]
            else:
                raise NotImplementedError
            for i in range(theta.size):
                x[i, :, 0] = r * np.sin(theta[i]) * np.cos(phi[i]) * self.res / 2
                x[i, :, 1] = r * np.sin(theta[i]) * np.sin(phi[i]) * self.res / 2
                x[i, :, 2] = r * np.cos(theta[i]) * self.depth / 2
            x = torch.from_numpy(x).float()
        elif initialization == 'gaussian':
            x = torch.randn(num_shots, num_measurements_shot, 3)
            x[:,:, 0] = x[:,:, 0] * self.res / 2
            x[:,:, 1] = x[:,:, 1] * self.res / 2
            x[:,:, 2] = x[:,:, 2] * self.depth / 2
        elif initialization == 'uniform':
            x = torch.rand(num_shots, num_measurements_shot, 3)*2-1
            x[:,:, 0] = x[:,:, 0] * self.res / 2
            x[:,:, 1] = x[:,:, 1] * self.res / 2
            x[:,:, 2] = x[:,:, 2] * self.depth / 2
        elif initialization == 'fullkspace':
            x = np.array(np.meshgrid(np.arange(-self.depth / 2, self.depth / 2), np.arange(-self.res / 2, self.res / 2),
                                     np.arange(-self.res / 2, self.res / 2))).T.reshape(num_shots,-1, 3)
            x = torch.from_numpy(x).float()
        elif initialization == '2dradial':
            num_sticks_slice = num_shots//self.depth
            x = np.zeros((num_sticks_slice, num_measurements_shot, 2))
            theta = np.pi / num_sticks_slice
            L = torch.arange(-self.res / 2, self.res / 2, self.res / num_measurements_shot).float()
            for i in range(num_sticks_slice):
                x[i, :, 0] = L * np.cos(theta * i)
                x[i, :, 1] = L * np.sin(theta * i)
            self.depthvec=torch.zeros((num_sticks_slice*self.depth,num_measurements_shot,1)).to('cuda')
            start=0
            for d in range(-self.depth//2,self.depth//2):
                self.depthvec[start:start+num_sticks_slice,:,:] = d if d==0 else (d-0.5 if d>0 else (d+0.5))
                start = start+num_sticks_slice
            x = torch.from_numpy(x).float()
        elif initialization == 'stackofstars' or initialization=='2dstackofstars':
            num_sticks_slice = num_shots//self.depth
            x = np.zeros((num_sticks_slice, num_measurements_shot, 2))
            theta = np.pi / num_sticks_slice
            L = torch.arange(-self.res / 2, self.res / 2, self.res / num_measurements_shot).float()
            for i in range(num_sticks_slice):
                x[i, :, 0] = L * np.cos(theta * i)
                x[i, :, 1] = L * np.sin(theta * i)
            depthvec=torch.zeros((num_sticks_slice*self.depth,num_measurements_shot,1))
            start=0
            for d in range(-self.depth//2,self.depth//2):
                depthvec[start:start+num_sticks_slice,:,:] = d if d==0 else (d-0.5 if d>0 else (d+0.5))
                start = start+num_sticks_slice
            x = torch.from_numpy(x).float()
            x = x.repeat(self.depth, 1, 1)
            x = torch.cat((x, depthvec), 2)
        else:
            print('Wrong initialization')
        self.orig_trajectory=x.data.to('cuda')
        if initialization=='2dradial':
            self.orig_trajectory=self.orig_trajectory.repeat(self.depth,1,1)
            self.orig_trajectory=torch.cat((self.orig_trajectory,self.depthvec),2)
        self.x = torch.nn.Parameter(x, requires_grad=trajectory_learning)
        return

    def __init__(self, acceleration_factor, res,depth, trajectory_learning,initialization,points_per_shot):
        super().__init__()

        self.acceleration_factor=acceleration_factor
        self.res=res
        self.depth=depth
        self.initialization=initialization
        self.points_per_shot=points_per_shot
        self.initilaize_trajectory(trajectory_learning, initialization,points_per_shot)

    def forward(self, input):
        input = input.squeeze(1).squeeze(1)

        x = self.x
        if self.initialization=='2dradial':
            x=x.repeat(self.depth,1,1)
            x=torch.cat((x,self.depthvec),2)
        x = x.reshape(-1, 3)
        x = x.clamp(-self.res / 2, self.res / 2)
        x=x.masked_scatter(x!=x,self.orig_trajectory)
        ksp = interp.bilinear_interpolate_torch_gridsample_3d(input, x)
        output= nufft.nufft_adjoint(ksp, x, input.shape, ndim=3)

        return output.unsqueeze(1)

    def get_trajectory(self):

        x=self.x
        if self.initialization=='2dradial':
            x=x.repeat(self.depth,1,1)
            x=torch.cat((x,self.depthvec),2)
        if self.training:
            return x
        x = x.data.cpu().numpy()
        scale = 3000 // self.points_per_shot
        X = np.zeros((x.shape[0] , x.shape[1]* scale,3))
        xs_old = np.linspace(0, self.points_per_shot, self.points_per_shot)
        xs_new = np.linspace(0, self.points_per_shot, self.points_per_shot * scale)
        for i in range(x.shape[0]):
            x_=interp1d(xs_old[x[i,:,0]==x[i,:,0]],x[i,:,0][x[i,:,0]==x[i,:,0]],kind='cubic')
            y_=interp1d(xs_old[x[i,:,1]==x[i,:,1]],x[i,:,1][x[i,:,1]==x[i,:,1]],kind='cubic')
            z_=interp1d(xs_old[x[i,:,2]==x[i,:,2]],x[i,:,2][x[i,:,2]==x[i,:,2]],kind='cubic')
            X[i, :, 0] = x_(xs_new)
            X[i, :, 1] = y_(xs_new)
            X[i, :, 2] = z_(xs_new)

        return torch.from_numpy(X).float()


    def __repr__(self):
        return f'Subsampling_Layer'

class Subsampling_Model_3D(nn.Module):
    def __init__(self, in_chans, out_chans, f_maps,acceleration_factor,res,depth, trajectory_learning,initialization,points_per_shot):
        super().__init__()

        self.subsampling=Subsampling_Layer_3D(acceleration_factor, res, depth, trajectory_learning,initialization,points_per_shot)
        self.reconstruction_model=UNet3D(in_chans,out_chans,True,f_maps=f_maps)


    def forward(self, input):
        input, _, __ = transforms.normalize_instance(input, eps=1e-11)
        input=self.subsampling(input)
        output = self.reconstruction_model(input)
        return output

    def get_trajectory(self):
        return self.subsampling.get_trajectory()

    def get_corrupted(self,input):
        corrupted = self.subsampling(input)
        return corrupted
