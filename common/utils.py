import json

import h5py
import pathlib

def get_vel_acc(x):
    # calculate numerical derivatives of the trajectory
    v = (x[:,1:, :] - x[:,:-1, :])
    a = (v[:,1:, :] - v[:,:-1, :])
    return v, a

def save_reconstructions(reconstructions, out_dir):
    """
    Saves the reconstructions from a model into h5 files that is appropriate for submission
    to the leaderboard.

    Args:
        reconstructions (dict[str, np.array]): A dictionary mapping input filenames to
            corresponding reconstructions (of shape num_slices x height x width).
        out_dir (pathlib.Path): Path to the output directory where the reconstructions
            should be saved.
    """
    pathlib.Path(out_dir).mkdir(exist_ok=True)
    for fname, recons in reconstructions.items():
        with h5py.File(out_dir +'/'+ fname, 'w') as f:
            f.create_dataset('reconstruction', data=recons)


def tensor_to_complex_np(data):
    """
    Converts a complex torch tensor to numpy array.
    Args:
        data (torch.Tensor): Input data to be converted to numpy.

    Returns:
        np.array: Complex numpy version of data
    """
    data = data.numpy()
    return data[..., 0] + 1j * data[..., 1]