#!/usr/bin/env python3
import pytest
import torch
import numpy as np
import geodesics as geo

class TestUnwrap(object):

    @pytest.mark.parametrize('ndim', [1,2,3,4,5])
    def test_unwrap(self, ndim):
        # discontinuity is at +/- pi, so test around that. probably not the
        # most efficient way to generate angles, but ah well
        start_angle = torch.tensor([np.pi+np.pi/4])
        start = torch.cat([torch.cos(start_angle), torch.sin(start_angle)]).reshape((1,1,1,2))
        angle = torch.tensor([np.pi-np.pi/4])
        stop = torch.cat([torch.cos(angle), torch.sin(angle)]).reshape((1,1,1,2))
        line = geo.make_straight_line(start, stop, 10)
        angles = torch.atan2(line[...,1], line[...,0]).squeeze()
        target_angles = np.unwrap(angles)
        while angles.ndim < ndim:
            angles = angles.unsqueeze(0)
        for target_dim in list(range(ndim)) + list(range(-ndim, 0)):
            target_shape = [1] * ndim
            target_shape[target_dim] = 11
            angles = angles.view(*target_shape)
            if not np.allclose(geo.unwrap(angles, target_dim).squeeze(), target_angles):
                raise ValueError(f"Unwrap failed for ndim: {ndim} and target_dim: {target_dim}")
        for fail_dim in list(range(ndim, 2*ndim)) + list(range(-2*ndim, -ndim)):
            with pytest.raises(ValueError, match="dim must lie within"):
                geo.unwrap(angles, fail_dim)
