from typing import Callable, Dict, List, Optional, Sequence, Union
import copy
import numpy as np
import os

import mmengine
from mmengine.config import Config, ConfigDict

from easycapture.registry import RUNNERS
from easycapture.transforms import Compose

ConfigType = Union[Dict, Config, ConfigDict]


@RUNNERS.register_module()
class Runner:
    def __init__(
        self,
        save_dir: str,
        pipeline: List[Union[dict, Callable]],
        cfg: Optional[ConfigType] = None,
    ):
        super().__init__()
        
        # save dir
        self._save_dir = save_dir
        mmengine.mkdir_or_exist(self._save_dir)
        
        # index
        self._frame_index = 0
        
        # pipeline
        self.pipeline = Compose(pipeline)
        
        # recursively copy the `cfg` because `self.cfg` will be modified
        # everywhere.
        if cfg is not None:
            if isinstance(cfg, Config):
                self.cfg = copy.deepcopy(cfg)
            elif isinstance(cfg, dict):
                self.cfg = Config(cfg)
        else:
            self.cfg = Config(dict())

    @property
    def save_dir(self) -> str:
        """str: The saving directory."""
        return self._save_dir
    
    @property
    def frame_index(self) -> int:
        """int: Current Frame Index."""
        return self._frame_index
    
    @classmethod
    def from_cfg(cls, cfg: ConfigType) -> 'Runner':
        """Build a runner from config.

        Args:
            cfg (ConfigType): A config used for building runner. Keys of
                ``cfg`` can see :meth:`__init__`.

        Returns:
            Runner: A runner build from ``cfg``.
        """
        cfg = copy.deepcopy(cfg)
        runner = cls(
            save_dir=cfg['save_dir'],
            pipeline=cfg['pipeline'],
            cfg=cfg,
        )
        return runner

    def run(self, meta_info: Optional[dict] = {}) -> None:
        """Launch Runner."""
        # run
        results = dict(meta_info=meta_info)
        results = self.pipeline(results)
        save_file = results['img_frame']
        
        # save
        if isinstance(save_file, np.ndarray):
            np.save(
                os.path.join(self._save_dir, '{:03d}.npy'.format(self._frame_index)), 
                save_file)
        else:
            raise NotImplementedError
        
        # frame index
        self._frame_index += 1
        