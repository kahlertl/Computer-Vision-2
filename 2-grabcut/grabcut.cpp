/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"
#include "gcgraph.hpp"
#include "grabcut.hpp"
#include <iostream>
#include <limits>

#ifndef NDEBUG
    #include <opencv2/highgui/highgui.hpp>
#endif

using namespace cv;

/**
 * This is implementation of image segmentation algorithm GrabCut described in
 * "GrabCut — Interactive Foreground Extraction using Iterated Graph Cuts".
 * Carsten Rother, Vladimir Kolmogorov, Andrew Blake.
 */

/**
 * GMM - Gaussian Mixture Model
 */
class GMM
{
  public:
    static const int componentsCount = 5;

    GMM(Mat &_model);

    double operator()(const Vec3d color) const;

    double operator()(int ci, const Vec3d color) const;

    int whichComponent(const Vec3d color) const;

    void initLearning();

    void addSample(int ci, const Vec3d color);

    void endLearning();

  private:
    void calcInverseCovAndDeterm(int ci);

    Mat model;
    double *coefs;
    double *mean;
    double *cov;

    double inverseCovs[componentsCount][3][3];
    double covDeterms[componentsCount];

    double sums[componentsCount][3];
    double prods[componentsCount][3][3];
    int sampleCounts[componentsCount];
    int totalSampleCount;
};

GMM::GMM(Mat &_model)
{
    // The model is single big array (a matrix with only 1 row) with the following
    // layout for all parameters:
    //
    // K = componentsCount
    //
    //        K              3K                          9K
    //
    //  1 coefficient   1 mean of each        1 coveriance matrix with 9
    //  per component   RGB channel per       values per component
    //                  component
    // +---------------------------------------------------------------------------+
    // |             |                     |                                       |
    // +---------------------------------------------------------------------------+
    //  ^             ^                      ^
    //  |             |                      |
    // coefs /       mean                   cov
    // weight
    //

    // number of parameters for each component of the model
    const int modelSize =   1  // component weight
                          + 3  // mean
                          + 9; // covariance
    if (_model.empty()) {
        _model.create(1, modelSize * componentsCount, CV_64FC1);
        _model.setTo(Scalar(0));
    } else if ((_model.type() != CV_64FC1) || (_model.rows != 1) || (_model.cols != modelSize * componentsCount)) {
        CV_Error(CV_StsBadArg, "_model must have CV_64FC1 type, rows == 1 and cols == 13*componentsCount");
    }

    model = _model;

    // Here comes some mad pointer arithmetic! See above for the matrix layout.
    coefs = model.ptr<double>(0); // gets pointer to the first (and only) row of the matrix
    mean = coefs + componentsCount;
    cov = mean + 3 * componentsCount;

    for (int ci = 0; ci < componentsCount; ci++) {
        if (coefs[ci] > 0) {
            calcInverseCovAndDeterm(ci);
        }
    }
}


/**
 * Returns the probability for a given color
 */
double GMM::operator()(const Vec3d color) const
{
    double res = 0;
    for (int ci = 0; ci < componentsCount; ci++) {
        res += coefs[ci] * (*this)(ci, color);
    }
    return res;
}

double GMM::operator()(int ci, const Vec3d color) const
{
    double res = 0;

    // if the coefficient for this component is 0, we can just skip
    // the computation
    if (coefs[ci] > 0) {
        CV_Assert(covDeterms[ci] > std::numeric_limits<double>::epsilon());
        Vec3d diff = color;
        double *m = mean + 3 * ci;
        diff[0] -= m[0];
        diff[1] -= m[1];
        diff[2] -= m[2];
        double mult =   diff[0] * (diff[0] * inverseCovs[ci][0][0] + diff[1] * inverseCovs[ci][1][0] +
                                   diff[2] * inverseCovs[ci][2][0])
                      + diff[1] * (diff[0] * inverseCovs[ci][0][1] + diff[1] * inverseCovs[ci][1][1] +
                                   diff[2] * inverseCovs[ci][2][1])
                      + diff[2] * (diff[0] * inverseCovs[ci][0][2] + diff[1] * inverseCovs[ci][1][2] +
                                   diff[2] * inverseCovs[ci][2][2]);
        res = 1.0f / sqrt(covDeterms[ci]) * exp(-0.5f * mult);
    }
    return res;
}

int GMM::whichComponent(const Vec3d color) const
{
    int k = 0;
    double max = 0;

    for (int ci = 0; ci < componentsCount; ci++) {
        double p = (*this)(ci, color);
        if (p > max) {
            k = ci;
            max = p;
        }
    }
    return k;
}

void GMM::initLearning()
{
    // reset all calculated sums and products to 0
    for (int ci = 0; ci < componentsCount; ci++) {
        sums[ci][0] = sums[ci][1] = sums[ci][2] = 0;
        prods[ci][0][0] = prods[ci][0][1] = prods[ci][0][2] = 0;
        prods[ci][1][0] = prods[ci][1][1] = prods[ci][1][2] = 0;
        prods[ci][2][0] = prods[ci][2][1] = prods[ci][2][2] = 0;
        sampleCounts[ci] = 0;
    }
    totalSampleCount = 0;
}

void GMM::addSample(int ci, const Vec3d color)
{
    sums[ci][0] += color[0];
    sums[ci][1] += color[1];
    sums[ci][2] += color[2];
    prods[ci][0][0] += color[0] * color[0];
    prods[ci][0][1] += color[0] * color[1];
    prods[ci][0][2] += color[0] * color[2];
    prods[ci][1][0] += color[1] * color[0];
    prods[ci][1][1] += color[1] * color[1];
    prods[ci][1][2] += color[1] * color[2];
    prods[ci][2][0] += color[2] * color[0];
    prods[ci][2][1] += color[2] * color[1];
    prods[ci][2][2] += color[2] * color[2];
    sampleCounts[ci]++;
    totalSampleCount++;
}

void GMM::endLearning()
{
    const double variance = 0.01;
    for (int ci = 0; ci < componentsCount; ci++) {
        int n = sampleCounts[ci];
        if (n == 0) {
            coefs[ci] = 0;
        } else {
            coefs[ci] = (double) n / totalSampleCount;

            double *m = mean + 3 * ci;
            m[0] = sums[ci][0] / n;
            m[1] = sums[ci][1] / n;
            m[2] = sums[ci][2] / n;

            double *c = cov + 9 * ci;
            c[0] = prods[ci][0][0] / n - m[0] * m[0];
            c[1] = prods[ci][0][1] / n - m[0] * m[1];
            c[2] = prods[ci][0][2] / n - m[0] * m[2];
            c[3] = prods[ci][1][0] / n - m[1] * m[0];
            c[4] = prods[ci][1][1] / n - m[1] * m[1];
            c[5] = prods[ci][1][2] / n - m[1] * m[2];
            c[6] = prods[ci][2][0] / n - m[2] * m[0];
            c[7] = prods[ci][2][1] / n - m[2] * m[1];
            c[8] = prods[ci][2][2] / n - m[2] * m[2];

            double dtrm = c[0] * (c[4] * c[8] - c[5] * c[7]) - c[1] * (c[3] * c[8] - c[5] * c[6]) +
                          c[2] * (c[3] * c[7] - c[4] * c[6]);
            if (dtrm <= std::numeric_limits<double>::epsilon()) {
                // Adds the white noise to avoid singular covariance matrix.
                c[0] += variance;
                c[4] += variance;
                c[8] += variance;
            }

            calcInverseCovAndDeterm(ci);
        }
    }
}

void GMM::calcInverseCovAndDeterm(int ci)
{
    if (coefs[ci] > 0) {
        double *c = cov + 9 * ci;
        double dtrm =
            covDeterms[ci] = c[0] * (c[4] * c[8] - c[5] * c[7]) - c[1] * (c[3] * c[8] - c[5] * c[6]) +
                             c[2] * (c[3] * c[7] - c[4] * c[6]);

        CV_Assert(dtrm > std::numeric_limits<double>::epsilon());
        inverseCovs[ci][0][0] =  (c[4] * c[8] - c[5] * c[7]) / dtrm;
        inverseCovs[ci][1][0] = -(c[3] * c[8] - c[5] * c[6]) / dtrm;
        inverseCovs[ci][2][0] =  (c[3] * c[7] - c[4] * c[6]) / dtrm;
        inverseCovs[ci][0][1] = -(c[1] * c[8] - c[2] * c[7]) / dtrm;
        inverseCovs[ci][1][1] =  (c[0] * c[8] - c[2] * c[6]) / dtrm;
        inverseCovs[ci][2][1] = -(c[0] * c[7] - c[1] * c[6]) / dtrm;
        inverseCovs[ci][0][2] =  (c[1] * c[5] - c[2] * c[4]) / dtrm;
        inverseCovs[ci][1][2] = -(c[0] * c[5] - c[2] * c[3]) / dtrm;
        inverseCovs[ci][2][2] =  (c[0] * c[4] - c[1] * c[3]) / dtrm;
    }
}

/**
 * Returns the number of edges in a graph of the image with
 * the given neighbors / connectivity.
 */
static inline double countEdges(const Mat &img, int neighbors)
{
    if (neighbors == GC_N4) {
        return   2 * img.cols * img.rows // each pixel has 2 neighbors (left, up)
               - img.cols                // the first colum has no left 
               - img.rows                // the first row has no up
               + 1;                      // we removed the upper left corner twice
    } else {
        return   4 * img.cols * img.rows // each pixel has 4 neighbors (left, upleft, up, upright)
               - 3 * img.cols            // the first colum has no left and upleft and the last colum
                                         // has no upright
               - 3 * img.rows            // the first row has only left neigbors
               + 2;                      // we removed the upper left and upper right corner twice
    }
}

/**
 * Calculate beta - parameter of GrabCut algorithm.
 *
 * beta = 1 / (2 * avg(sqr(||color[i] - color[j]||)))
 */
static double calcBeta(const Mat &img, int neighbors)
{
    double beta = 0;
    for (int y = 0; y < img.rows; y++) {
        for (int x = 0; x < img.cols; x++) {
            Vec3d color = img.at<Vec3b>(y, x);
            // left
            if (x > 0) {
                Vec3d diff = color - (Vec3d) img.at<Vec3b>(y, x - 1);
                beta += diff.dot(diff);
            }
            // upleft
            if (neighbors == GC_N8 && y > 0 && x > 0) {
                Vec3d diff = color - (Vec3d) img.at<Vec3b>(y - 1, x - 1);
                beta += diff.dot(diff);
            }
            // up
            if (y > 0) {
                Vec3d diff = color - (Vec3d) img.at<Vec3b>(y - 1, x);
                beta += diff.dot(diff);
            }
            // upright
            if (neighbors == GC_N8 && y > 0 && x < img.cols - 1) {
                Vec3d diff = color - (Vec3d) img.at<Vec3b>(y - 1, x + 1);
                beta += diff.dot(diff);
            }
        }
    }
    if (beta <= std::numeric_limits<double>::epsilon()) {
        return 0;
    } else {
        return 1.f / (2 * beta / countEdges(img, neighbors));
    }
}

/**
 * Calculate beta for the extended pairwise / binary / smoothing term 
 * as parameter of GrabCut algorithm.
 *
 * beta = 2 / (avg(||color[i] - color[j]||))
 */
static double calcExtendedBeta(const Mat &img, int neighbors)
{
    double beta = 0;

    for (int y = 0; y < img.rows; y++) {
        for (int x = 0; x < img.cols; x++) {
            Vec3d color = img.at<Vec3b>(y, x);
            // left
            if (x > 0) {
                beta += norm(color - (Vec3d) img.at<Vec3b>(y, x - 1));
            }
            // upleft
            if (neighbors == GC_N8 && y > 0 && x > 0) {
                beta += norm(color - (Vec3d) img.at<Vec3b>(y - 1, x - 1));
            }
            // up
            if (y > 0) {
                beta += norm(color - (Vec3d) img.at<Vec3b>(y - 1, x));
            }
            // upright
            if (neighbors == GC_N8 && y > 0 && x < img.cols - 1) {
                beta += norm(color - (Vec3d) img.at<Vec3b>(y - 1, x + 1));
            }
        }
    }
    if (beta <= std::numeric_limits<double>::epsilon()) {
        return 0;
    }

    return 2.f / (beta / countEdges(img, neighbors));
}

static void calcExtendedNWeights(const Mat &img, Mat &leftW, Mat &upleftW, Mat &upW, Mat &uprightW,
                                 double beta, double gamma, int neighbors)
{
    CV_Error(CV_StsNotImplemented,"not implemented yet");
}

/**
 * Calculate weights of noterminal vertices of graph.
 * N means the neighbors of the graph.
 * 
 * beta and gamma - parameters of GrabCut algorithm.
 */
static void calcNWeights(const Mat &img, Mat &leftW, Mat &upleftW, Mat &upW, Mat &uprightW,
                         double beta, double gamma, int neighbors)
{
    const double gammaDivSqrt2 = gamma / std::sqrt(2.0f);

    // initialize all matrices
    leftW.create(img.rows, img.cols, CV_64FC1);
    upleftW.create(img.rows, img.cols, CV_64FC1);
    upW.create(img.rows, img.cols, CV_64FC1);
    uprightW.create(img.rows, img.cols, CV_64FC1);

    for (int y = 0; y < img.rows; y++) {
        for (int x = 0; x < img.cols; x++) {
            Vec3d color = img.at<Vec3b>(y, x);
            // left
            if (x - 1 >= 0) {
                Vec3d diff = color - (Vec3d) img.at<Vec3b>(y, x - 1);
                leftW.at<double>(y, x) = gamma * exp(-beta * diff.dot(diff));
            } else {
                leftW.at<double>(y, x) = 0;
            }
            // upleft
            if (neighbors == GC_N8 && x - 1 >= 0 && y - 1 >= 0) {
                Vec3d diff = color - (Vec3d) img.at<Vec3b>(y - 1, x - 1);
                upleftW.at<double>(y, x) = gammaDivSqrt2 * exp(-beta * diff.dot(diff));
            } else {
                upleftW.at<double>(y, x) = 0;
            }
            // up
            if (y - 1 >= 0) {
                Vec3d diff = color - (Vec3d) img.at<Vec3b>(y - 1, x);
                upW.at<double>(y, x) = gamma * exp(-beta * diff.dot(diff));
            } else {
                upW.at<double>(y, x) = 0;
            }
            // upright
            if (neighbors == GC_N8 && x + 1 < img.cols && y - 1 >= 0) {
                Vec3d diff = color - (Vec3d) img.at<Vec3b>(y - 1, x + 1);
                uprightW.at<double>(y, x) = gammaDivSqrt2 * exp(-beta * diff.dot(diff));
            } else {
                uprightW.at<double>(y, x) = 0;
            }
        }
    }
}

/**
 * Check size, type and element values of mask matrix.
 */
static void checkMask(const Mat &img, const Mat &mask)
{
    if (mask.empty()) {
        CV_Error(CV_StsBadArg, "mask is empty");
    }
    if (mask.type() != CV_8UC1) {
        CV_Error(CV_StsBadArg, "mask must have CV_8UC1 type");
    }
    if (mask.cols != img.cols || mask.rows != img.rows) {
        CV_Error(CV_StsBadArg, "mask must have as many rows and cols as img");
    }
    for (int y = 0; y < mask.rows; y++) {
        for (int x = 0; x < mask.cols; x++) {
            uchar val = mask.at<uchar>(y, x);
            if (val != GC_BGD && val != GC_FGD && val != GC_PR_BGD && val != GC_PR_FGD)
                CV_Error(CV_StsBadArg, "mask element value must be equel"
                    "GC_BGD or GC_FGD or GC_PR_BGD or GC_PR_FGD");
        }
    }
}

/**
 * Initialize mask using rectangular.
 */
static void initMaskWithRect(Mat &mask, Size imgSize, Rect rect)
{
    // set the whole mask to background
    mask.create(imgSize, CV_8UC1);
    mask.setTo(GC_BGD);

    // sanitize rectangular dimensions
    rect.x = max(0, rect.x);
    rect.y = max(0, rect.y);
    rect.width = min(rect.width, imgSize.width - rect.x);
    rect.height = min(rect.height, imgSize.height - rect.y);

    // set the whole inner area of the rectangular to
    // "probably background"
    (mask(rect)).setTo(Scalar(GC_PR_FGD));
}

/**
 * Initialize GMM background and foreground models using kmeans algorithm.
 */
static void initGMMs(const Mat &img, const Mat &mask, double tolerance,
                     GMM &bgdGMM, GMM &fgdGMM)
{
    const int kMeansItCount = 10;
    const int kMeansType = KMEANS_PP_CENTERS;

    Mat bgdLabels, fgdLabels;
    vector<Vec3f> bgdSamples, fgdSamples;
    vector<Point> bgdPixels, fgdPixels;
    Point p;
    for (p.y = 0; p.y < img.rows; p.y++) {
        for (p.x = 0; p.x < img.cols; p.x++) {
            if (mask.at<uchar>(p) == GC_BGD || mask.at<uchar>(p) == GC_PR_BGD) {
                bgdSamples.push_back((Vec3f) img.at<Vec3b>(p));
                bgdPixels.push_back(p);
            } else { // GC_FGD | GC_PR_FGD
                fgdSamples.push_back((Vec3f) img.at<Vec3b>(p));
                fgdPixels.push_back(p);
            }
        }
    }
    // cluster input data with k-means algorithm
    CV_Assert(!bgdSamples.empty() && !fgdSamples.empty());
    Mat _bgdSamples((int) bgdSamples.size(), 3, CV_32FC1, &bgdSamples[0][0]);
    kmeans(_bgdSamples, GMM::componentsCount, bgdLabels,
           TermCriteria(CV_TERMCRIT_ITER, kMeansItCount, 0.0), 0, kMeansType);
    Mat _fgdSamples((int) fgdSamples.size(), 3, CV_32FC1, &fgdSamples[0][0]);
    kmeans(_fgdSamples, GMM::componentsCount, fgdLabels,
           TermCriteria(CV_TERMCRIT_ITER, kMeansItCount, 0.0), 0, kMeansType);

    bgdGMM.initLearning();
    for (int i = 0; i < (int) bgdSamples.size(); i++) {
        bgdGMM.addSample(bgdLabels.at<int>(i, 0), bgdSamples[i]);
    }
    bgdGMM.endLearning();

    // determine the set of foreground samples that are most unlikly in the
    // background dsitribution.
    vector<double> probabilities(fgdSamples.size());
    vector<int>    sampleIdx(fgdSamples.size());
    for (int i = 0; i < (int) fgdSamples.size(); i++) {
        probabilities[i] = bgdGMM(fgdSamples[i]);
        sampleIdx[i] = i;
    }
    std::sort(sampleIdx.begin(), sampleIdx.end(), [&probabilities] (int const& a, int const& b) {
        return probabilities[a] < probabilities[b];
    });

    #ifndef NDEBUG
        Mat canvas;
        img.copyTo(canvas);
    #endif

    fgdGMM.initLearning();
    for (int i = 0; i < sampleIdx.size() * tolerance; i++) {
        #ifndef NDEBUG
            cv::circle(canvas, fgdPixels[sampleIdx[i]], 2, cv::Scalar(0, 0, 255), -1);
        #endif

        fgdGMM.addSample(fgdLabels.at<int>(sampleIdx[i], 0), fgdSamples[sampleIdx[i]]);
    }
    fgdGMM.endLearning();

    #ifndef NDEBUG
        // display points used for foreground distribution
        cv::imshow("Foreground selection", canvas);
    #endif
}

/**
 * Assign GMMs components for each pixel.
 */
static void assignGMMsComponents(const Mat &img, const Mat &mask, const GMM &bgdGMM, const GMM &fgdGMM, Mat &compIdxs)
{
    Point p;
    for (p.y = 0; p.y < img.rows; p.y++) {
        for (p.x = 0; p.x < img.cols; p.x++) {
            Vec3d color = img.at<Vec3b>(p);
            compIdxs.at<int>(p) = mask.at<uchar>(p) == GC_BGD || mask.at<uchar>(p) == GC_PR_BGD ?
                                  bgdGMM.whichComponent(color) : fgdGMM.whichComponent(color);
        }
    }
}

/**
 * Learn GMMs parameters.
 */
static void learnGMMs(const Mat &img, const Mat &mask, const Mat &compIdxs, GMM &bgdGMM, GMM &fgdGMM)
{
    bgdGMM.initLearning();
    fgdGMM.initLearning();
    Point p;
    for (int ci = 0; ci < GMM::componentsCount; ci++) {
        for (p.y = 0; p.y < img.rows; p.y++) {
            for (p.x = 0; p.x < img.cols; p.x++) {
                if (compIdxs.at<int>(p) == ci) {
                    if (mask.at<uchar>(p) == GC_BGD || mask.at<uchar>(p) == GC_PR_BGD) {
                        bgdGMM.addSample(ci, img.at<Vec3b>(p));
                    } else {
                        fgdGMM.addSample(ci, img.at<Vec3b>(p));
                    }
                }
            }
        }
    }
    bgdGMM.endLearning();
    fgdGMM.endLearning();
}

/**
 * Construct GCGraph
 * 
 * A graph is built from the GMM. Nodes in the graphs are pixels.
 * Two additional nodes are added, the Source node and the Sink node. Every foreground
 * pixel is connected to Source node and every background pixel is connected to Sink node.
 * 
 * The weights of edges connecting pixels to source/sink node node are defined by the
 * probability of a pixel being foreground/background.
 * 
 * The weights between the pixels are defined by the edge information or pixel similarity.
 * If there is a large difference in pixel color, the edge between them will get a low weight.
 * 
 * The graph is 8-connected, which means that every pixel/node is connected to each of
 * it neighbors.
 * 
 * @param lambda    weight of the edges connecting foreground/background nodes
 *                  to the source/sink
 */
static void constructGCGraph(const Mat &img, const Mat &mask, const GMM &bgdGMM, const GMM &fgdGMM,
                             double lambda, int neighbors,
                             const Mat &leftW, const Mat &upleftW, const Mat &upW, const Mat &uprightW,
                             GCGraph<double> &graph)
{
    int vtxCount = img.cols * img.rows;
    int edgeCount = 2 * (4 * img.cols * img.rows - 3 * (img.cols + img.rows) + 2);
    graph.create(vtxCount, edgeCount);
    Point p;
    for (p.y = 0; p.y < img.rows; p.y++) {
        for (p.x = 0; p.x < img.cols; p.x++) {
            // add node
            int vtxIdx = graph.addVtx();
            Vec3b color = img.at<Vec3b>(p);

            // 
            // Unary / data term
            // 
            // set t-weights
            double fromSource, toSink;
            // we do not know exactly if the pixel is fore- or background, therefore
            // it is not connected to either the source or the sink 
            if (mask.at<uchar>(p) == GC_PR_BGD || mask.at<uchar>(p) == GC_PR_FGD) {
                fromSource = -log(bgdGMM(color));
                toSink     = -log(fgdGMM(color));
            }
            // background pixels are all connected to the sink
            else if (mask.at<uchar>(p) == GC_BGD) {
                fromSource = 0;
                toSink = lambda;
            }
            // foreground pixels are all connected to the source
            else {
                fromSource = lambda;
                toSink = 0;
            }
            graph.addTermWeights(vtxIdx, fromSource, toSink);

            // 
            // Pairwise / binary / smoothing term
            // 
            // set n-weights
            if (p.x > 0) {
                double w = leftW.at<double>(p);
                graph.addEdges(vtxIdx, vtxIdx - 1, w, w);
            }
            if (neighbors == GC_N8 && p.x > 0 && p.y > 0) {
                double w = upleftW.at<double>(p);
                graph.addEdges(vtxIdx, vtxIdx - img.cols - 1, w, w);
            }
            if (p.y > 0) {
                double w = upW.at<double>(p);
                graph.addEdges(vtxIdx, vtxIdx - img.cols, w, w);
            }
            if (neighbors == GC_N8 && p.x < img.cols - 1 && p.y > 0) {
                double w = uprightW.at<double>(p);
                graph.addEdges(vtxIdx, vtxIdx - img.cols + 1, w, w);
            }
        }
    }
}

/**
 * Estimate segmentation using MaxFlow algorithm
 */
static void estimateSegmentation(GCGraph<double> &graph, Mat &mask)
{
    graph.maxFlow();
    Point p;
    for (p.y = 0; p.y < mask.rows; p.y++) {
        for (p.x = 0; p.x < mask.cols; p.x++) {
            if (mask.at<uchar>(p) == GC_PR_BGD || mask.at<uchar>(p) == GC_PR_FGD) {
                if (graph.inSourceSegment(p.y * mask.cols + p.x /*vertex index*/)) {
                    mask.at<uchar>(p) = GC_PR_FGD;
                } else {
                    mask.at<uchar>(p) = GC_PR_BGD;
                }
            }
        }
    }
}

void cv::grabCut(InputArray _img, InputOutputArray _mask, Rect rect,
                 InputOutputArray _bgdModel, InputOutputArray _fgdModel,
                 int iterCount, double tolerance, bool extended,
                 double connectivity, double contrast,
                 int neighbors, int mode)
{
    Mat img = _img.getMat();
    Mat &mask = _mask.getMatRef();
    Mat &bgdModel = _bgdModel.getMatRef();
    Mat &fgdModel = _fgdModel.getMatRef();

    if (img.empty()) {
        CV_Error(CV_StsBadArg, "image is empty");
    }
    if (img.type() != CV_8UC3) {
        CV_Error(CV_StsBadArg, "image must have CV_8UC3 type");
    }

    GMM bgdGMM(bgdModel), fgdGMM(fgdModel);
    Mat compIdxs(img.size(), CV_32SC1);

    if (mode == GC_INIT_WITH_RECT || mode == GC_INIT_WITH_MASK) {
        if (mode == GC_INIT_WITH_RECT) {
            initMaskWithRect(mask, img.size(), rect);
        } else { // flag == GC_INIT_WITH_MASK
            checkMask(img, mask);
        }
        initGMMs(img, mask, tolerance, bgdGMM, fgdGMM);
    }

    if (iterCount <= 0) {
        return;
    }

    // if we do not initialize the mask, check it for consistency
    if (mode == GC_EVAL) {
        checkMask(img, mask);
    }

    const double gamma = 50;
    const double lambda = 9 * gamma;

    Mat leftW, upleftW, upW, uprightW;
    
    if (extended) {
        const double beta = calcExtendedBeta(img, neighbors);
        calcExtendedNWeights(img, leftW, upleftW, upW, uprightW, beta, gamma, neighbors);
    } else {
        const double beta = calcBeta(img, neighbors);
        calcNWeights(img, leftW, upleftW, upW, uprightW, beta, gamma, neighbors);
    }


    for (int i = 0; i < iterCount; i++) {
        GCGraph<double> graph;
        assignGMMsComponents(img, mask, bgdGMM, fgdGMM, compIdxs);
        learnGMMs(img, mask, compIdxs, bgdGMM, fgdGMM);
        constructGCGraph(img, mask, bgdGMM, fgdGMM, lambda, neighbors,
                         leftW, upleftW, upW, uprightW, graph);
        estimateSegmentation(graph, mask);
    }
}
