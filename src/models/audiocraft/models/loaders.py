# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Utility functions to load from the checkpoints.
Each checkpoint is a torch.saved dict with the following keys:
- 'xp.cfg': the hydra config as dumped during training. This should be used
    to rebuild the object using the audiocraft.models.builders functions,
- 'model_best_state': a readily loadable best state for the model, including
    the conditioner. The model obtained from `xp.cfg` should be compatible
    with this state dict. In the case of a LM, the encodec model would not be
    bundled along but instead provided separately.

Those functions also support loading from a remote location with the Torch Hub API.
They also support overriding some parameters, in particular the device and dtype
of the returned model.
"""

from pathlib import Path
from huggingface_hub import hf_hub_download
import typing as tp
import os

from omegaconf import OmegaConf
import torch

from . import builders


HF_MODEL_CHECKPOINTS_MAP = {
    "small": "facebook/musicgen-small",
    "medium": "facebook/musicgen-medium",
    "large": "facebook/musicgen-large",
    "melody": "facebook/musicgen-melody",
}


def _get_state_dict(
    file_or_url_or_id: tp.Union[Path, str],
    filename: tp.Optional[str] = None,
    device='cpu',
    cache_dir: tp.Optional[str] = None,
):
    # Return the state dict either from a file or url
    file_or_url_or_id = str(file_or_url_or_id)
    assert isinstance(file_or_url_or_id, str)

    cached_compression = '/Users/bytedance/.cache/huggingface/hub/models--facebook--musicgen-melody/snapshots/51e18546f12df76256cdf55199560a82366389fe/compression_state_dict.bin'
    cached_state = '/Users/bytedance/.cache/huggingface/hub/models--facebook--musicgen-melody/snapshots/51e18546f12df76256cdf55199560a82366389fe/state_dict.bin'

    if os.path.isfile(cached_compression) or os.path.isfile(cached_state):
        print('读取缓存的模型')
        if filename == "compression_state_dict.bin":
            print('加载compression_state_dict.bin模型')
            return torch.load(cached_compression, map_location=device)
        elif filename == "state_dict.bin":
            print('加载state_dict.bin模型')
            return torch.load(cached_state, map_location=device)
        print('没有读到缓存的模型')

    elif file_or_url_or_id.startswith('https://'):
        print('从指定远程url中拉取模型:', file_or_url_or_id)
        return torch.hub.load_state_dict_from_url(file_or_url_or_id, map_location=device, check_hash=True)

    elif file_or_url_or_id in HF_MODEL_CHECKPOINTS_MAP:
        print('从huggingface中下载模型')
        assert filename is not None, "filename needs to be defined if using HF checkpoints"

        repo_id = HF_MODEL_CHECKPOINTS_MAP[file_or_url_or_id]
        file = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=cache_dir)
        return torch.load(file, map_location=device)

    else:
        raise ValueError(f"{file_or_url_or_id} is not a valid name, path or link that can be loaded.")


def load_compression_model(file_or_url_or_id: tp.Union[Path, str], device='cpu', cache_dir: tp.Optional[str] = None):
    pkg = _get_state_dict(file_or_url_or_id, filename="compression_state_dict.bin", cache_dir=cache_dir)
    cfg = OmegaConf.create(pkg['xp.cfg'])
    cfg.device = str(device)
    model = builders.get_compression_model(cfg)
    model.load_state_dict(pkg['best_state'])
    model.eval()
    return model


def load_lm_model(file_or_url_or_id: tp.Union[Path, str], device='cpu', cache_dir: tp.Optional[str] = None):
    pkg = _get_state_dict(file_or_url_or_id, filename="state_dict.bin", cache_dir=cache_dir)
    cfg = OmegaConf.create(pkg['xp.cfg'])
    cfg.device = str(device)
    if cfg.device == 'cpu':
        cfg.dtype = 'float32'
    else:
        cfg.dtype = 'float16'
    model = builders.get_lm_model(cfg)
    model.load_state_dict(pkg['best_state'])
    model.eval()
    model.cfg = cfg
    return model
