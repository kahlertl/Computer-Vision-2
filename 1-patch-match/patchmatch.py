#!/usr/bin/env python2

import numpy as np
import cv2
from visual import parser, load_frames, show_flow
import random
from itertools import product


def ssd(image1, center1, image2, center2, size):
    pass


class PatchMatch(object):
    """Dump implementation of the PatchMatch algorithm as described by

    Connelly Barnes, Eli Shechtman, Adam Finkelstein, and Dan B. Goldman.
    PatchMatch: A randomized correspondence algorithm for structural image editing.
    In ACM Transactions on Graphics (Proc. SIGGRAPH), 2009. 2
    """

    def __init__(self, image1, image2):
        self.image1 = image1
        self.image2 = image2

        self.nrows = self.image1.shape[0]
        self.ncols = self.image1.shape[1]

        # create an empty matrix with the same x-y dimensions like the first
        # image but with two channels. Each channel stands for an x/y offset
        # of a pixel at this position.
        self.result  = np.empty(dtype=np.int16, shape=(self.nrows, self.ncols, 2))
        self.quality = np.empty(dtype=np.float32, shape=(self.nrows, self.ncols))

        # initialize offsets randomly
        self.initialize()

    def __iter__(self):
        for index in product(xrange(self.nrows), xrange(self.ncols)):
            yield index

    def initialize(self):
        for index in self:
            # create a random offset in 
            offset = random.randint(-20, 20), random.randint(-20, 20)

            # assing random offset
            self.result[index] = offset

            # calculate the center in the second image by adding the offset
            # to the current index
            center = index[0] + offset[0], index[1] + offset[1]

            self.quality[index] = ssd(self.image1, index, self.image2, center, 5)

    def iterate(self):
        for index in self:
            offset = self.result[index]
            self.propagate(index, offset)

    def propagate(self, index, offset):
        pass

    def random_search(self):
        pass


if __name__ == '__main__':
    # command line parsing
    args = parser.parse_args()
    frame1, frame2 = load_frames(args.image1, args.image2)

    print('initialize ...')
    pm = PatchMatch(frame1, frame2)

    print('iterate ...')
    pm.iterate()

    # # do some iterations
    # for i in xrange(3):
    #     pm.iterate()

    # display final result
    # we have to convert the integer offsets to floats, because
    # optical flow could be subpixel accurate
    show_flow(np.float32(pm.result))
