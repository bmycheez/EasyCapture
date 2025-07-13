import os
import argparse
import cv2
import numpy as np

from easycapture.transforms import Compose


def parse_args():
    parser = argparse.ArgumentParser(description='Raw Data to RGB Image')
    parser.add_argument('raw_path', help='raw image path')
    parser.add_argument(
        '--show', action='store_true', help='show rgb image')
    parser.add_argument(
        '--save-dir',
        help='directory where rgb images will be saved. ')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    # pipeline
    PIPELINE = [
        dict(type='LoadNumpyArray'),
        dict(type='BlackWhiteLevel', black_level=240, white_level=4095),
        dict(type='Bayer2RGB', bayer='gbrg'),
        dict(type='AstypeNumpy', input_type='uint16', output_type='float'),
        dict(type='AutoWhiteBalance'),
        dict(type='ColorCorrectionMatrix', ccm=[[1.8, -0.8, 0], [-0.3, 1.5, -0.2], [0, -0.8, 1.8]]),
        dict(type='GammaCorrection', digital_gain=0.1, gamma=2.2),
        dict(type='AstypeNumpy', input_type='float', output_type='uint8'),
        dict(type='CvtColor', input_type='rgb', output_type='bgr')]
    
    pipeline = Compose(PIPELINE)
    
    # load
    raw_path = args.raw_path
    results = dict(img_path = raw_path)
    results = pipeline(results)
    bgr_image = results['img_frame']
    
    # show
    if args.show:
        cv2.imshow('RGB Image', bgr_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if args.save_dir is not None:
        save_name = os.path.splitext(os.path.basename(args.raw_path))[0] + '.png'
        cv2.imwrite(os.path.join(args.save_dir, save_name), bgr_image)

    
if __name__ == '__main__':
    main()