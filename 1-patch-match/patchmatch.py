#!/usr/bin/env python2

import numpy as np
import cv2
from visual import load_frames, flow2rgb, show
from similarity import ssd
import random
from itertools import product
import argparse
import sys


def echo(*args, **kwargs):
    end = kwargs.get('end', '\n')

    if len(args) == 1:
        sys.stdout.write(str(args[0]))
    else:
        for arg in args:
            sys.stdout.write(str(arg))
            sys.stdout.write(' ')

    sys.stdout.write(end)
    sys.stdout.flush()


class PatchMatch(object):
    """Dump implementation of the PatchMatch algorithm as described by

    Connelly Barnes, Eli Shechtman, Adam Finkelstein, and Dan B. Goldman.
    PatchMatch: A randomized correspondence algorithm for structural image editing.
    In ACM Transactions on Graphics (Proc. SIGGRAPH), 2009. 2
    """

    def __init__(self, image1, image2, match_radius=5, search_ratio=0.5, search_radius=None, maxoffset=10):
        if not 0 < search_ratio < 1:
            raise ValueError('Search ratio must be in interval (0,1)')

        # input
        self.image1 = image1
        self.image2 = image2
        self.nrows, self.ncols = self.image1.shape

        self.niterations = 0

        # parameters
        self.maxoffset = maxoffset
        self.match_radius = match_radius
        self.search_ratio = search_ratio
        self.search_radius = search_radius or maxoffset

        # TODO: only match radius as border
        self.border = self.match_radius

        # create an empty matrix with the same x-y dimensions like the first
        # image but with two channels. Each channel stands for an x/y offset
        # of a pixel at this position.
        self.result  = np.zeros(dtype=np.float32, shape=(self.nrows, self.ncols, 2))


    def __iter__(self):
        rows = xrange(self.border, self.nrows - self.border)
        cols = xrange(self.border, self.ncols - self.border)

        for index in product(rows, cols):
            yield index


    def in_border(self, center):
        return  self.border < center[0] < self.nrows - self.border and \
                self.border < center[1] < self.ncols - self.border


    def initialize(self, prior_knowledge=None):
        # use precomputed offsets and qualities
        if not prior_knowledge is None:
            # expand prior knowledge to size of result array
            # result_shape = np.shape(self.result)[0]
            # prior_shape = np.shape(prior_knowledge)[0]
            # ratio = float(result_shape) / float(prior_shape)
            # echo(result_shape, "/", prior_shape, "=", ratio)

            self.result = cv2.resize(prior_knowledge, (self.ncols, self.nrows))
            flow = flow2rgb(self.result)
            show(flow, wait=True)


            # self.result = prior_knowledge
        else:
            for index in self:
                # create a random offset in 
                # TODO: check offset is inside image

                offset = 0

                while (True):
                    offset = random.randint(-self.maxoffset, self.maxoffset), random.randint(-self.maxoffset, self.maxoffset)
                    center = index[0] + offset[0], index[1] + offset[1]

                    # check if the center is inside the other image otherwise choose new offset
                    if self.in_border(center):
                        break

                # assing random offset
                
                self.result[index] = offset


    def iterate(self):
        self.niterations += 1

        # switch between top and left neighbor in even iterations and
        # right bottom neighbor in odd iterations
        self.neighbor = -1 if self.niterations % 2 == 0 else 1

        rows = xrange(self.border, self.nrows - self.border)
        cols = xrange(self.border, self.ncols - self.border)

        for row in rows:
            echo("\r%d" % row, end='')
            for col in cols:
                index = row, col
                # echo('index', index, end='')
                self.propagate(index)
                self.random_search(index)
            echo("\r", end='')


    def propagate(self, index):
        indices = (index,                               # current position
                  (index[0] + self.neighbor, index[1]), # top / bottom neighbor
                  (index[0], index[1] + self.neighbor)) # left / right neighbor
    
        # create an array of all qualities at the above indices
        qualities = np.empty([3])

        # calculate the quality of the current pixel with the offsets of the neighboring ones
        for i, neighbor in enumerate(indices):
            center = index + self.result[neighbor]
            if self.in_border(center):
                qualities[i] = ssd(self.image1, index, self.image2, center, self.match_radius)
            else:
                qualities[i] = float("inf")
        
        # get the index of the best quality (smallest distance)
        minindex = indices[np.argmin(qualities)]

        # get the offset from the neighbor with the best quality
        if minindex != index:
            self.result[index] = self.result[minindex]


    def random_search(self, index):
        i = 0
        offset = self.result[index]
        best_quality = ssd(self.image1, index, self.image2, index + offset, self.match_radius)

        while True:
            distance = self.search_radius * self.search_ratio ** i
            i += 1

            # halt condition. search radius must not be smaller
            # than one pixel
            if distance < 1:
                break

            choice = random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)

            new_offset_x = (int) (offset[0] + distance * choice[0])
            new_offset_y = (int) (offset[1] + distance * choice[1])
            new_offset = new_offset_x, new_offset_y

            center  = index[0] + new_offset_x, index[1] + new_offset_y

            # check that we do not jump outside the image
            if self.in_border(center):
                quality = ssd(self.image1, index, self.image2, center, self.match_radius)

                if quality < best_quality:
                    # check that the new offset is not greater than the maximum offset
                    if abs(new_offset[0]) > self.maxoffset and abs(new_offset[1]) > self.maxoffset:
                        continue

                    self.result[index] = new_offset
                    best_quality = quality


def reconstruct_from_flow(flow, image):
    result = np.zeros_like(image)

    for index in np.ndindex(flow.shape[0], flow.shape[1]):
        offset = flow[index]
        pixel  = index[0] + offset[0], index[1] + offset[1]
        result[index] = image[pixel]

    return result


def merge(image1, image2):
    # convert images into RGB if they are grayscale
    if len(image1.shape) == 2:
        image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
    if len(image2.shape) == 2:
        image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)

    height1, width1 = image1.shape[:2]
    height2, width2 = image2.shape[:2]

    canvas = np.zeros((max(height1, height2), width1 + width2, 3), dtype=np.uint8)
    canvas[ : height1, : width1] = image1
    canvas[ : height2, width1 : width1 + width2] = image2

    return canvas


parser = argparse.ArgumentParser()
parser.add_argument('image1', help="First frame")
parser.add_argument('image2', help="Second frame")
parser.add_argument('-i', '--iterations', type=int, default=3)
parser.add_argument('--match-radius', type=int, default=4)
parser.add_argument('--search-ratio', type=float, default=0.5)
parser.add_argument('--search-radius', type=int, default=None)
parser.add_argument('--maxoffset', type=int, default=15)
parser.add_argument('--pyramid', '-p', type=int, default=1)

if __name__ == '__main__':
    try:
        # command line parsing
        args = parser.parse_args()
        frame1, frame2 = load_frames(args.image1, args.image2)

        print('Parameters:')
        print('  iterations:    %d' % args.iterations)
        print('  match-radius:  %d' % args.match_radius)
        print('  maxoffset:     %d' % args.maxoffset)
        print('  search-radius: %d' % (args.search_radius or min(frame1.shape[:2])))
        print('  search-ratio:  %f' % args.search_ratio)
        print('  pyramid depth: %d' % args.pyramid)
        print('')

        pyramid = [(frame1, frame2)]

        for i in xrange(args.pyramid - 1):
            image1, image2 = pyramid[-1]
            pyramid.append((cv2.resize(image1, (0,0), fx=0.5, fy=0.5), cv2.resize(image2, (0,0), fx=0.5, fy=0.5)))

        for index, images in enumerate(reversed(pyramid)):
            pm = PatchMatch(images[0], images[1],
                        match_radius=args.match_radius, search_ratio=args.search_ratio,
                        maxoffset=args.maxoffset, search_radius=args.search_radius)

            print('initialize ...')
            # initialize offsets randomly
            if index == 0:
                pm.initialize()
            # initialize offsets with previous pyramid result
            else:
                pm.initialize(pyramid_result)

            flow = flow2rgb(pm.result)
            reconstruction = reconstruct_from_flow(pm.result, frame2)
            merged = merge(flow, reconstruction)
            cv2.imwrite('flow-it-0.png', merged)

            # do some iterations
            for i in xrange(args.iterations):
                print('iteration %d ...' % (i + 1))
                pm.iterate()

                # display progress
                # we have to convert the integer offsets to floats, because
                # optical flow could be subpixel accurate
                flow = flow2rgb(pm.result)
                reconstruction = reconstruct_from_flow(pm.result, frame2)
                merged = merge(flow, reconstruction) 
                # show(merged)
                name = 'flow-it-%i.png' % (i + 1)
                cv2.imwrite(name, merged)

            print("result of patchmatch in pyramid step: %d" % (index + 1))
            pyramid_result = pm.result
            flow = flow2rgb(pyramid_result)
            reconstruction = reconstruct_from_flow(pyramid_result, frame2)
            show(merge(flow, reconstruction), wait=True)

    except KeyboardInterrupt:
        print('Stopping ...')
