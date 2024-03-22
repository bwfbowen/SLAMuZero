#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from ..core.registry import registry
from .eqa.mp3d_eqa_dataset import Matterport3dDatasetV1
from .pointnav.pointnav_dataset import PointNavDatasetV1


def make_dataset(id_dataset, **kwargs):
    _dataset = registry.get_dataset(id_dataset)
    assert _dataset is not None, "Could not find dataset {}".format(id_dataset)

    return _dataset(**kwargs)
