#!/usr/bin/env python2

import numpy as np
import cv2
from visual import parser, load_frames

class PatchMatch(object):
    """Dump implementation of the PatchMatch algorithm as described by

    Connelly Barnes, Eli Shechtman, Adam Finkelstein, and Dan B. Goldman.
    Patchmatch: A randomized correspondence algorithm for structural image editing.
    In ACM Transactions on Graphics (Proc. SIGGRAPH), 2009. 2
    """

    def initialize(self):
        pass

    def iterate(self):
        pass

    def propagate(self):
        pass

    def random_search(self):
        pass


if __name__ == '__main__':
    args = parser.parse_args()
    frame1, frame2 = load_frames(args.image1, args.image2)
