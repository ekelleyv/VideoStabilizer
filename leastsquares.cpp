#include "opencv2/opencv.hpp"
#include "matching.h"
#include <math.h>
#include <stdio.h>

using namespace cv;

#define PI 3.14159

void best_transform(const CvSeq* imageKeypointsA, const CvSeq* imageKeypointsB, double origin_x, double origin_y, const vector<int> ptpairs, double *distance, double *angle, double *lambdacostheta, double *lambdasintheta, double *Tx, double *Ty) {
  Mat b(ptpairs.size(), 1, CV_64F);
  Mat A(ptpairs.size(), 4, CV_64F);

  for (int i= 0; i < ptpairs.size(); i+=2) {
    CvSURFPoint* pt_a = (CvSURFPoint*)cvGetSeqElem( imageKeypointsA, ptpairs[i]);
    CvSURFPoint* pt_b = (CvSURFPoint*)cvGetSeqElem( imageKeypointsB, ptpairs[i+1]);
    b.at<double>(i,0) = pt_b->pt.x - origin_x;
    b.at<double>(i+1,0) = pt_b->pt.y - origin_y;
    A.at<double>(i, 0) = pt_a->pt.x - origin_x;
    A.at<double>(i, 1) = -(pt_a->pt.y - origin_y);
    A.at<double>(i, 2) = 1;
    A.at<double>(i, 3) = 0;
    A.at<double>(i+1, 0) = pt_a->pt.y - origin_y;
    A.at<double>(i+1, 1) = pt_a->pt.x - origin_x;
    A.at<double>(i+1, 2) = 0;
    A.at<double>(i+1, 3) = 1;
  }
  
  /*
  for (int i=0; i < ptpairs.size(); i++) {
    printf("%f, %f, %f, %f\n", A.at<double>(i,0), A.at<double>(i,1), A.at<double>(i,2), A.at<double>(i,3));
  }
  */
  

  /*
  for (int i=0; i < ptpairs.size(); i++) {
    printf("%f\n", b.at<double>(i,0));
  }
  */
  

  Mat x = (A.t()*A).inv()*(A.t()*b);
  *lambdacostheta = x.at<double>(0, 0);
  *lambdasintheta = x.at<double>(1, 0);
  *Tx = x.at<double>(2, 0);
  *Ty = x.at<double>(3, 0);
  //*lambda = sqrt(lambdacostheta*lambdacostheta+lambdasintheta*lambdasintheta);
  //*theta = atan(lambdasintheta/lambdacostheta);

  for (int i=0; i < ptpairs.size(); i+=2) {
    CvSURFPoint* pt_a = (CvSURFPoint*)cvGetSeqElem(imageKeypointsA, ptpairs[i]);
    CvSURFPoint* pt_b = (CvSURFPoint*)cvGetSeqElem(imageKeypointsB, ptpairs[i+1]);
    double expected_x = (pt_a->pt.x - origin_x)*(*lambdacostheta)-(pt_a->pt.y - origin_y)*(*lambdasintheta)+*Tx + origin_x;
    double expected_y = (pt_a->pt.x - origin_x)*(*lambdasintheta)+(pt_a->pt.y - origin_y)*(*lambdacostheta)+*Ty + origin_y;
    double delta_x = pt_b->pt.x-expected_x;
    double delta_y = pt_b->pt.y-expected_y;
    distance[i/2] = sqrt(delta_x*delta_x + delta_y*delta_y);
    double match_angle = atan2(pt_b->pt.y-pt_a->pt.y, pt_b->pt.x-pt_a->pt.x);
    double expected_angle = atan2(expected_y-pt_a->pt.y, expected_x-pt_a->pt.x);
    double angle_diff = match_angle - expected_angle;    
    if (angle_diff > PI) {
      angle_diff = angle_diff-(2*PI);
    }
    else if (angle_diff < -PI) {
      angle_diff = angle_diff+(2*PI);
    }
    angle[i/2] = fabs(angle_diff);
  }
  
}
