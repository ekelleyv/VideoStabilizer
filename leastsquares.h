#ifndef test_leastsquares_h
#define test_leastsquares_h

#include "opencv2/opencv.hpp"

using namespace cv;

void best_transform(const CvSeq* imageKeypointsA, const CvSeq* imageKeypointsB, double origin_x, double origin_y, const vector<int> ptpairs, double *distance, double *angle, double *lambda, double *theta, double *Tx, double *Ty);

#endif
