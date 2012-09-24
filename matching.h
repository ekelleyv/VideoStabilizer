//
//  matching.h
//  test
//
//  Created by Minqi Jiang on 1/12/12.
//  Copyright 2012 Princeton University. All rights reserved.
//

#ifndef test_matching_h
#define test_matching_h

#include "opencv2/opencv.hpp"

using namespace cv;


// this function takes two sets of feature descriptors and stores the set of
// matchings in a vector of ptpairs
void
flannFindPairs( const CvSeq *imageDescriptorsA,
			    const CvSeq *imageDescriptorsB, vector<int>& ptpairs );

// given image keypoints in two images and corresponding pair indices, computes
// local displacement vectors between pairs and saves in a vector points
void 
findLocalDisplacements( const CvSeq *imageKeypointsA, const CvSeq *imageKeypointsB,
                       vector<int>& ptpairs, vector<CvPoint2D32f>& localDisplacements);

void
showKeypoints( IplImage *image, const CvSeq *imageKeypoints );

void showArrows(IplImage *image, const CvSeq *imageKeypointsA, const CvSeq *imageKeypointsB, vector<int> ptpairs);


#endif
