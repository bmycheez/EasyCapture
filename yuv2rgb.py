import argparse
import cv2
from struct import *
import os

from easycapture.transforms import Compose


def parse_args():
    parser = argparse.ArgumentParser(prog='YUV2RGB Converter', description='Convert YUV Raw images into RGB format')
    parser.add_argument('--raw_path', type=str, help='raw image path')
    parser.add_argument('--width', type=int, default=1920)
    parser.add_argument('--height', type=int, default=1080)
    parser.add_argument('--frame', default=0, type=int, help='frame number to grab (default: index 0)')
    parser.add_argument('--version', action='version', version='%(prog)s 1.0')
    parser.add_argument(
        '--show', action='store_true', help='show rgb image')
    parser.add_argument(
        '--save-dir',
        help='directory where rgb images will be saved. ')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    if args.frame < 0:
        args.frame = 0

    PIPELINE = [
        dict(type='NV12toRGB',
             width=args.width,
             height=args.height,
             frame=args.frame),
    ]

    pipeline = Compose(PIPELINE)

    # load
    raw_path = args.raw_path
    results = dict(img_path = raw_path)
    bgr_image = pipeline(results)

    # show
    if args.show:
        cv2.imshow('RGB Image', bgr_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if args.save_dir is not None:
        save_name = os.path.splitext(os.path.basename(args.raw_path))[0] + '.png'
        cv2.imwrite(os.path.join(args.save_dir, save_name), bgr_image)

if __name__=='__main__':
    main()
