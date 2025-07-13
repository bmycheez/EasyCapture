import os
import os.path as osp
import argparse
import time
import subprocess

import cv2
import numpy as np

from mmengine.config import Config, DictAction
from easycapture.registry import RUNNERS
from easycapture.runners import Runner


def parse_args():
    parser = argparse.ArgumentParser(description='Data Capture')
    parser.add_argument('config', help='capture config path')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # set camera
    gain = cfg['camera']['gain']
    exposure = cfg['camera']['exposure'] * 1000
    frame_rate = int(1e9 / cfg['camera']['fps'])
    frame_cnt = cfg['frame_cnt']
    
    subprocess.call(["v4l2-ctl", "-d", "0", "-c", f"frame_rate={frame_rate}"])
    subprocess.call(["v4l2-ctl", "-d", "0", "-c", f"exposure={exposure},gain={gain}"])
    time.sleep(2)
    
    for _ in range(frame_cnt):
        # run
        runner.run()
        
        # break
        cmd = cv2.waitKey(1)
        if cmd == ord("q"):
            break

if __name__ == "__main__":
    main()