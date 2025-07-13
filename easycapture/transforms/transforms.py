from typing import List, Optional, Sequence, Tuple, Union, Any

import cv2
import numpy as np
import os
import math

from easycapture.registry import TRANSFORMS
from .base import BaseTransform


@TRANSFORMS.register_module()
class GetRawFramefromCamera(BaseTransform):
    """Get a RAW Frame from Camera.

    This pipeline get a RAW frame image from Camera and
    returns the frame.

    Added Keys:

    - img_frame

    Args:
        device_num (int): Camera device number.
        size (tuple[int]): (w, h)
    """
    def __init__(self,
                 device_num: int = 0,
                 size: Tuple[int, int] = (1920, 1080)) -> None:
        self.device_num = device_num
        self.size = size
        self._setting_cam()  # setting cam

    def _setting_cam(self) -> None:
        """Setting Camera."""
        self.cam = cv2.VideoCapture(self.device_num, cv2.CAP_V4L2)
        self.cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"GB12"))
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])

    def _postprocess(self, frame: np.ndarray) -> np.ndarray:
        """Postprocess the frame obtained from camera.

        Args:
            frame (np.ndarray): A frame obtained from camera.

        Returns:
            np.ndarray: A postprocessed frame. (h, w)
        """
        # flatten frame
        frame_flatten = frame.flatten()

        # uint16 to uint12
        raw_16 = frame_flatten.view(np.uint16)[: self.size[0] * self.size[1]]
        raw_12 = raw_16 >> 4

        # reshape
        raw_reshape = raw_12.reshape(self.size[1], self.size[0])
        return raw_reshape

    def transform(self, results: dict) -> dict:
        """Transform function to get a RAW frame from camera.

        Args:
            results (dict): Result dict.

        Returns:
            dict: The result dict contains the frame taken from camera.
        """
        _, frame = self.cam.read()  # read frame from camera
        frame = self._postprocess(frame)  # postprocess
        results['img_frame'] = frame
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(device_num={self.device_num}, '
        repr_str += f'size={self.size})'
        return repr_str


@TRANSFORMS.register_module()
class LoadNumpyArray(BaseTransform):
    def __init__(self) -> None:
        pass

    def transform(self, results: dict) -> dict:
        img_path = results['img_path']
        img_frame = np.load(img_path)
        results['img_frame'] = img_frame
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@TRANSFORMS.register_module()
class BlackWhiteLevel(BaseTransform):
    def __init__(self,
                 black_level: int = 240, 
                 white_level: int = 4095) -> None:
        self.black_level = black_level
        self.white_level = white_level

    def transform(self, results: dict) -> dict:
        img_frame = results['img_frame']  # W H, np.uint16
        img_frame = img_frame.astype(np.float64)
        
        # black white level
        img_frame = (img_frame - self.black_level) / \
            (self.white_level - self.black_level)  # black level
        img_frame = np.clip(img_frame, 0, 1)  # clip
        
        # float64 to uint16
        img_frame = img_frame * 65535
        img_frame = img_frame.astype(np.uint16)
        results['img_frame'] = img_frame
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(black_level={self.black_level}, '
        repr_str += f'white_level={self.white_level})'
        return repr_str


@TRANSFORMS.register_module()
class Bayer2RGB(BaseTransform):
    def __init__(self,
                 bayer: str = 'gbrg') -> None:
        self.bayer = bayer

    def transform(self, results: dict) -> dict:
        img_frame = results['img_frame']  # W H, uint16
        
        if self.bayer == 'gbrg':
            img_frame = cv2.cvtColor(img_frame, cv2.COLOR_BayerGR2RGB)
        elif self.bayer == 'bggr':
            img_frame = cv2.cvtColor(img_frame, cv2.COLOR_BayerRG2RGB)
        elif self.bayer == 'rggb':
            img_frame = cv2.cvtColor(img_frame, cv2.COLOR_BayerBG2RGB)
        else:
            raise NotImplementedError

        results['img_frame'] = img_frame
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__ + f'(bayer={self.bayer})'
        return repr_str
    

@TRANSFORMS.register_module()
class AstypeNumpy(BaseTransform):
    def __init__(self,
                 input_type: str,
                 output_type: str) -> None:
        self.input_type = input_type
        self.output_type = output_type

    def transform(self, results: dict) -> dict:
        img_frame = results['img_frame']
        if self.input_type == 'uint16' and self.output_type == 'float':
            img_frame = img_frame.astype(np.float64)
            img_frame /= 65535
        elif self.input_type =='float' and self.output_type == 'uint8':
            img_frame *= 255
            img_frame = img_frame.astype(np.uint8)
        else:
            raise NotImplementedError
        results['img_frame'] = img_frame
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(input_type={self.input_type}, '
        repr_str += f'output_type={self.output_type})'
        return repr_str


@TRANSFORMS.register_module()
class CvtColor(BaseTransform):
    def __init__(self,
                 input_type: str,
                 output_type: str) -> None:
        self.input_type = input_type
        self.output_type = output_type

    def transform(self, results: dict) -> dict:
        img_frame = results['img_frame']
        if self.input_type == 'bgr' and self.output_type == 'rgb':
            img_frame = cv2.cvtColor(img_frame, cv2.COLOR_BGR2RGB)
        elif self.input_type =='rgb' and self.output_type == 'bgr':
            img_frame = cv2.cvtColor(img_frame, cv2.COLOR_RGB2BGR)
        else:
            raise NotImplementedError
        results['img_frame'] = img_frame
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(input_type={self.input_type}, '
        repr_str += f'output_type={self.output_type})'
        return repr_str
    

@TRANSFORMS.register_module()
class AutoWhiteBalance(BaseTransform):
    def __init__(self) -> None:
        pass

    def transform(self, results: dict) -> dict:
        img_frame = results['img_frame']
        r = img_frame[:, :, 0] * (np.mean(img_frame[:, :, 1]) / np.mean(img_frame[:, :, 0]))
        g = img_frame[:, :, 1]
        b = img_frame[:, :, 2] * (np.mean(img_frame[:, :, 1]) / np.mean(img_frame[:, :, 2]))
        img_frame = np.stack([r, g, b], axis=2)
        img_frame = np.clip(img_frame, 0, 1)
        results['img_frame'] = img_frame
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@TRANSFORMS.register_module()
class ColorCorrectionMatrix(BaseTransform):
    def __init__(self, ccm: list) -> None:
        self.ccm = ccm

    def transform(self, results: dict) -> dict:
        img_frame = results['img_frame']
        r = img_frame[:, :, 0]
        g = img_frame[:, :, 1]
        b = img_frame[:, :, 2]
        r_ccm = float(self.ccm[0][0]) * r + float(self.ccm[0][1]) * g + float(self.ccm[0][2]) * b
        g_ccm = float(self.ccm[1][0]) * r + float(self.ccm[1][1]) * g + float(self.ccm[1][2]) * b
        b_ccm = float(self.ccm[2][0]) * r + float(self.ccm[2][1]) * g + float(self.ccm[2][2]) * b
        img_frame = np.stack([r_ccm, g_ccm, b_ccm], axis=2)
        img_frame = np.clip(img_frame, 0, 1)
        results['img_frame'] = img_frame
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__ + f'(ccm={self.ccm})'
        return repr_str


@TRANSFORMS.register_module()
class GammaCorrection(BaseTransform):
    def __init__(self, 
                 digital_gain: float,
                 gamma: float) -> None:
        self.digital_gain = digital_gain
        self.gamma = gamma

    def transform(self, results: dict) -> dict:
        img_frame = results['img_frame']
        # digital gain
        img_frame = img_frame * (float(self.digital_gain) / np.mean(img_frame))
        img_frame = np.clip(img_frame, 0, 1)
        # gamma correction
        img_frame = img_frame ** (1 / float(self.gamma))
        img_frame = np.clip(img_frame, 0, 1)
        results['img_frame'] = img_frame
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(digital_gain={self.digital_gain}, '
        repr_str += f'gamma={self.gamma})'
        return repr_str


@TRANSFORMS.register_module()
class NV12toRGB(BaseTransform):
    def __init__(self,
                 width: int = 1920,
                 height: int = 1080,
                 frame: int = 0) -> None:
        self.width = width
        self.height = height
        self.frame = frame

    def transform(self, results: dict) -> dict:
        img_path = results['img_path']
        f_y = open(img_path, "rb")
        f_uv= open(img_path, "rb")
        
        pixels = np.zeros((self.height, self.width, 3))
        
        size_of_file = os.path.getsize(img_path)
        size_of_frame = ((3.0/2.0)*self.height*self.width)
        number_of_frames = size_of_file / size_of_frame
        frame_start = size_of_frame * self.frame
        uv_start = frame_start + (self.width*self.height)
        
 		#lets get our y cursor ready
        f_y.seek(int(frame_start));        
        for j in range(0, self.height):
            for i in range(0, self.width):
				#uv_index starts at the end of the yframe.  The UV is 1/2 height so we multiply it by j/2
				#We need to floor i/2 to get the start of the UV byte
                uv_index = uv_start + (self.width * math.floor(j/2)) + (math.floor(i/2))*2
                f_uv.seek(int(uv_index))
                
                y = ord(f_y.read(1))
                u = ord(f_uv.read(1))
                v = ord(f_uv.read(1))
				
                b = 1.164 * (y - 16) + 2.018 * (u - 128)
                g = 1.164 * (y - 16) - 0.813 * (v - 128) - 0.391 * (u - 128)
                r = 1.164 * (y - 16) + 1.596 * (v - 128)
                
                pixels[i,j] = int(r), int(g), int(b)
        
        results['img_frame'] = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str
