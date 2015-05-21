#!/usr/bin/env python2

import numpy as np
import cv2
import argparse

def ssd(image1, center1, image2, center2, size):
    window1 = image1[center1[0] - size : center1[0] + size,
                     center1[1] - size : center1[1] + size]

    window2 = image2[center2[0] - size : center2[0] + size,
                     center2[1] - size : center2[1] + size]

    # array substraction
    diff = window1 - window2

    return np.sum(diff ** 2)


parser = argparse.ArgumentParser()
parser.add_argument('image1')
parser.add_argument('image2')
parser.add_argument('--center1-x', type=int, default=3)
parser.add_argument('--center1-y', type=int, default=5)
parser.add_argument('--center2-x', type=int, default=10)
parser.add_argument('--center2-y', type=int, default=5)
parser.add_argument('--size', type=int, default=3)


if __name__=='__main__':
    args = parser.parse_args()

    print('Parameters:')
    print('  center in image1: (%d, %d)' % (args.center1_x, args.center1_y))
    print('  center in image2: (%d, %d)' % (args.center2_x, args.center2_y))
    print('  window size:      %d' % args.size)
    print('')

    # load an color image
    frame1 = cv2.imread(args.image1, 0)
    frame2 = cv2.imread(args.image2, 0)

    # center of patch in image1
    center1 = args.center1_x, args.center1_y
    center2 = args.center2_x, args.center2_y

    ssd_value = ssd(frame1, center1, frame2, center2, args.size)
    print('SSD result: %d' % ssd_value)