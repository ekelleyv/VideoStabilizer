//
// for feature matching
//
#include "opencv2/opencv.hpp"
#include "matching.h"
#include <math.h>

#define THRESHOLD 0.6
#define PI 3.14159

using namespace cv;

// this function takes two sets of feature descriptors and stores the set of
// matchings in a passed-in vector called ptpairs


void
flannFindPairs( const CvSeq* imageDescriptorsA,
			    const CvSeq* imageDescriptorsB, 
                vector<int>& ptpairs)
{

	int length = (int)(imageDescriptorsA->elem_size/sizeof(float));
	
        cv::Mat m_imageA(imageDescriptorsA->total, length, CV_32F);
	cv::Mat m_imageB(imageDescriptorsB->total, length, CV_32F);
	
	//cout << "copying desciptors for flann...\n";
    // copy descriptors
    CvSeqReader img_readerA;
	float* img_ptrA = m_imageA.ptr<float>(0);
    cvStartReadSeq( imageDescriptorsA, &img_readerA );
    for(int i = 0; i < imageDescriptorsA->total; i++ )
    {
        const float* descriptor = (const float*)img_readerA.ptr;
        CV_NEXT_SEQ_ELEM( img_readerA.seq->elem_size, img_readerA );
        memcpy(img_ptrA, descriptor, length*sizeof(float));
        img_ptrA += length;
    }

    CvSeqReader img_readerB;
	float* img_ptrB = m_imageB.ptr<float>(0);
    cvStartReadSeq( imageDescriptorsB, &img_readerB );

    for(int i = 0; i < imageDescriptorsB->total; i++ )
    {
        const float* descriptor = (const float*)img_readerB.ptr;
        CV_NEXT_SEQ_ELEM( img_readerB.seq->elem_size, img_readerB );
        memcpy(img_ptrB, descriptor, length*sizeof(float));
        img_ptrB += length;
    }
	
    // find nearest neighbors using FLANN
    cv::Mat m_indices(imageDescriptorsA->total, 2, CV_32S);
    cv::Mat m_dists(imageDescriptorsA->total, 2, CV_32F);
    cv::flann::Index flann_index(m_imageA, cv::flann::KDTreeIndexParams(4));  // using 4 randomized kdtrees
    flann_index.knnSearch(m_imageB, m_indices, m_dists, 2, cv::flann::SearchParams(256) ); // maximum number of leafs checked is 256
	
    
	// only save pair if distance to 1st NN to 2nd NN is less than 0.6
    int* indices_ptr = m_indices.ptr<int>(0);
    float* dists_ptr = m_dists.ptr<float>(0);
    for (int i=0;i<m_indices.rows;++i) {
        if (dists_ptr[2*i] < THRESHOLD*dists_ptr[2*i+1]) {
            ptpairs.push_back(indices_ptr[2*i]);
            ptpairs.push_back(i);
        }
    }
}


void 
calcLocalDisplacements( const CvSeq *imageKeypointsA, const CvSeq *imageKeypointsB,
                       vector<int>& ptpairs, vector<CvPoint2D32f>& localDisplacements) 
{   
    for (int i = 0; i < ptpairs.size(); i+=2) 
    {
        CvSURFPoint* pt_a = (CvSURFPoint*)cvGetSeqElem( imageKeypointsA, ptpairs[2*i] );
        CvSURFPoint* pt_b = (CvSURFPoint*)cvGetSeqElem( imageKeypointsB, ptpairs[2*i+1] );
        localDisplacements.push_back(cvPoint2D32f(pt_b->pt.x - pt_a->pt.x, pt_b->pt.y - pt_a->pt.y));
    }
    return;
}

void
showKeypoints( IplImage *image, const CvSeq *imageKeypoints ) 
{
    IplImage *color_output = cvCreateImage(cvSize(image->width, image->height), 8, 3);
    cvCvtColor(image, color_output, CV_GRAY2RGB);
    
    CvSeqReader reader;
    cvStartReadSeq( imageKeypoints, &reader, 0 );
    for (int i = 0; i < imageKeypoints->total; i++) 
    {
        // get next keypoint
        const CvSURFPoint* kp = (const CvSURFPoint*)reader.ptr;
        CV_NEXT_SEQ_ELEM( reader.seq->elem_size, reader );
        
        // draw the keypoint
        CvPoint pt;
        pt.x = kp->pt.x;
        pt.y = kp->pt.y;
        cvCircle( color_output, pt, 3, CV_RGB(0,255,0), 3, 8, 0 );
    }
    cvNamedWindow("result", 0);
    cvShowImage("result", color_output );
    cvWaitKey(15);
}

void showArrows(IplImage *image, const CvSeq *imageKeypointsA, const CvSeq *imageKeypointsB, vector<int> ptpairs)
{
    IplImage *color_output = cvCreateImage(cvSize(image->width, image->height), 8, 3);
    cvCvtColor(image, color_output, CV_GRAY2RGB);

    for (int i = 0; i < ptpairs.size(); i+=2) {
        // get next keypoints
        const CvSURFPoint* kpA = (CvSURFPoint*)cvGetSeqElem(imageKeypointsA, ptpairs[i]);

        const CvSURFPoint* kpB = (CvSURFPoint*)cvGetSeqElem(imageKeypointsB, ptpairs[i+1]);

        
        // draw the keypoints
        CvPoint ptA;
        ptA.x = kpA->pt.x;
        ptA.y = kpA->pt.y;

        CvPoint ptB;
        ptB.x = kpB->pt.x;
        ptB.y = kpB->pt.y;

	double angle = atan2((double)ptA.y - ptB.y, (double) ptA.x - ptB.x);
	double distance = sqrt((ptA.y - ptB.y)*(ptA.y - ptB.y) + (ptA.x - ptB.x)*(ptA.x - ptB.x));


	// choose color and size of arrow
	int red = 0;
	int green = 0;
	int blue = 0;
	if (distance < 20) {
		distance = 20;
		green = 255;
	}
	else if (distance > 80) {
		distance = 80;
		red = 255;
	}
	else {
		red = 200;
		green = 200;
	}


	// draw the main line
	ptA.x = (int) ptB.x + distance*cos(angle);
	ptA.y = (int) ptB.y + distance*sin(angle);
	cvLine(color_output, ptA, ptB, CV_RGB(red,green,blue), 4, 8, 0);

	// draw the two sides of arrow tip at 50% length and 45 degrees away
	ptA.x = (int) ptB.x + distance*.5*cos(angle + PI/4);
	ptA.y = (int) ptB.y + distance*.5*sin(angle + PI/4);
	cvLine(color_output, ptA, ptB, CV_RGB(red,green,blue), 4, 8, 0);

	ptA.x = (int) ptB.x + distance*.5*cos(angle - PI/4);
	ptA.y = (int) ptB.y + distance*.5*sin(angle - PI/4);
	cvLine(color_output, ptA, ptB, CV_RGB(red,green,blue), 4, 8, 0);

    }
    cvNamedWindow("result", 0);
    cvShowImage("result", color_output );
    cvWaitKey(15);
}
