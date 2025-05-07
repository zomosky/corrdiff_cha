import datetime
import json
import math
from typing import List, Tuple, Union

import numpy as np
from numba import jit, prange
import xarray as xr

from modulus.utils.generative import convert_datetime_to_cftime

from .base import ChannelMetadata, DownscalingDataset


class SPrixinDataset(DownscalingDataset):
    """
    Reader for dataset used for corrdiff in .nc type
    """

    def __int__(
            self,
            data_path: str,
            stats_path: str,
            input_variable: Union[List[str], None, str] = None,
            output_variable: Union[List[str], None, str] = None,
            invariant_variable: Union[List[str], None, str] = None,
    ):
        """
        data_path: the path of .nc data of varibales 
        stats_path: the path of stats, include mean and std, reduce the compute cost
        input: the name of input variables
        output: the name of output varibales
        invariant: the name of invariant(such as elev and land_sea_mask)

        the input_channel = input + invariant
        the output_channel = output
        """
        self.data_path = data_path
        self.stats_path = stats_path
        self.input, self.input_variable = _load_dataset(
            self.data_path, input_variable, group=None
        )
        self.output, self.output_variable = _load_dataset(
            self.data_path, output_variable, group=None
        )
        self.invariant, self.invariant_variable = _load_dataset(
            self.data_path, invariant_variable, group=None
        )








    def __getitem__(self, idx):





    def __len__(self):


    """ Return longitude values form dataset. """
    def longitude(self) -> np.ndarray:


    """ Return latitude values from dataset """
    def latitude(self) -> np.ndarray:


    """ Return Metadata used for input train """
    def input_channels(self) -> List[ChannelMetadata]:


    """ Return Metadata used for ouput target """
    def output_channels(self) -> List[ChannelMetadata]:


    """ Return time values from the dataset """
    def time(self) -> List:



    """
    Return the shape of input data
    (same size of input and output in width and height)
    """
    def image_shape(self) -> Tuple[int, int]:


    """ Return normalized input data, used in load dataset"""
    def normalize_input(self, x: np.ndarray) -> np.ndarray:


    """ Return denormalized output, used in generate """
    def denormalize_input(self, x: np.ndarray) -> np.ndarray:


    """ Return normalized output data, used in load dataset"""
    def normalize_output(self, x: np.ndarray) -> np.ndarray:


    """ Return denormalized output, used in generate """
    def denormalize_output(self, x: np.ndarray) -> np.ndarray:


    """
    Return upsample input data by the shape of output data
    used _zoom_extrapolate
    """
    def upsample(self, x):
        y_shape = (
            x.shape[0],
            x.shape[1] * self.upsample_factor,
            x.shape[2] * self.upsample_factor,
        )
        y = np.empty(y_shape, dtype=np.float32)
        _zoom_extrapolate(x, y, self.upsample_factor)
        return y







""" Load nc dataset """
def _load_dataset(data_path, group=None, variables=None, stack_axis=1):
    if not (variables and stack_axis):
        raise ValueError("Either NC read group and variables must be provided, but not both")
    if group is None:
        with xr.open_dataset(data_path) as ds:
            if variables is None:
                variables = list(ds.keys())
            data = np.stack([ds[v] for v in variables], axis=stack_axis)
    else:
        with xr.open_dataset(data_path, group=group) as ds:
            if variables is None:
                variables = list(ds.keys())
            data = np.stack([ds[v] for v in variables], axis=stack_axis)
    return data, variables


""" Load stats of data, used to normalize and denormalize """
def _load_stats(stats, variables, group):
    mean = np.array([stats[group][v]["mean"] for v in variables])[:, None, None].astype(
        np.float32
    )
    std = np.array([stats[group][v]["std"] for v in variables])[:, None, None].astype(
        np.float32
    )
    return mean, std


@jit(nopython=True, parallel=True)
def _zoom_extrapolate(x, y, factor):
    """Bilinear zoom with extrapolation.
    Use a numba function here because numpy/scipy options are rather slow.

    Used in upsample images and variable fields
    """
    s = 1 / factor
    for k in prange(y.shape[0]):
        for iy in range(y.shape[1]):
            ix = (iy + 0.5) * s - 0.5
            ix0 = int(math.floor(ix))
            ix0 = max(0, min(ix0, x.shape[1] - 2))
            ix1 = ix0 + 1
            for jy in range(y.shape[2]):
                jx = (jy + 0.5) * s - 0.5
                jx0 = int(math.floor(jx))
                jx0 = max(0, min(jx0, x.shape[2] - 2))
                jx1 = jx0 + 1

                x00 = x[k, ix0, jx0]
                x01 = x[k, ix0, jx1]
                x10 = x[k, ix1, jx0]
                x11 = x[k, ix1, jx1]
                djx = jx - jx0
                x0 = x00 + djx * (x01 - x00)
                x1 = x10 + djx * (x11 - x10)
                y[k, iy, jy] = x0 + (ix - ix0) * (x1 - x0)
