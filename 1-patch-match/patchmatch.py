#!/usr/bin/env python2

import numpy as np
import cv2
from visual import parser, load_frames, show_flow
import random


class PatchMatch(object):
    """Dump implementation of the PatchMatch algorithm as described by

    Connelly Barnes, Eli Shechtman, Adam Finkelstein, and Dan B. Goldman.
    PatchMatch: A randomized correspondence algorithm for structural image editing.
    In ACM Transactions on Graphics (Proc. SIGGRAPH), 2009. 2
    """

    def __init__(self, image1, image2):
        self.image1 = image1
        self.image2 = image2

        # create an empty matrix with the same x-y dimensions like the first
        # image but with two channels. Each channel stands for an x/y offset
        # of a pixel at this position.
        self.result = np.empty(dtype=np.int16, shape=(image1.shape[0], image1.shape[1], 2))

        # initialize offsets randomly
        self.initialize()

    def initialize(self):
        for x in np.nditer(self.result, op_flags=['readwrite']):
            x[...] = random.randint(-20, 20)

    def iterate(self):
        pass

    def propagate(self):
        pass

    def random_search(self):
        pass


if __name__ == '__main__':
    # command line parsing
    args = parser.parse_args()
    frame1, frame2 = load_frames(args.image1, args.image2)

    pm = PatchMatch(frame1, frame2)

    # print(pm.result)

    # do some iterations
    for i in xrange(3):
        pm.iterate()

    # display final result
    # we have to convert the integer offsets to floats, because
    # optical flow could be subpixel accurate
    show_flow(np.float32(pm.result))
