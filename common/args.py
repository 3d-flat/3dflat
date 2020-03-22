import argparse
import pathlib


class Args(argparse.ArgumentParser):
    """
    Defines global default arguments.
    """

    def __init__(self, **overrides):
        """
        Args:
            **overrides (dict, optional): Keyword arguments used to override default argument values
        """

        super().__init__(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        self.add_argument('--seed', default=42, type=int, help='Seed for random number generators')
        self.add_argument('--resolution', default=320, type=int, help='Resolution of images')
        self.add_argument('--depth', default=16, type=int, help='Depth of image (number of slices to be included in 3d volume)')
        self.add_argument('--resolution_degrading', default=1, type=int, help='Size of Kernel which is used to degrade the resolution.'
                                                                              'Will degrade with kernel and then crop down to resolution.')

        # Data parameters
        self.add_argument('--data-path', type=pathlib.Path,
                          default='/home/tomerweiss/Datasets',help='Path to the dataset')
        self.add_argument('--sample-rate', type=float, default=1.,
                          help='Fraction of total volumes to include')


        # Override defaults with passed overrides
        self.set_defaults(**overrides)
