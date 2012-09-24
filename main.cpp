#include "opencv2/opencv.hpp"
#include "matching.h"
#include "leastsquares.h"
#include "fuzzylogic.h"
#include <stdio.h>

using namespace cv;
using namespace std;

#define PI 3.14159

double returnITF(char *filename) {

	CvCapture* cap = cvCaptureFromAVI(filename);
	IplImage* frameA = 0;
	IplImage* frameB = 0;

    frameB = cvQueryFrame(cap);
    frameA = cvCloneImage(frameB);
	int frames = 0;
	double PNSR = 0;
	while (frameB = cvQueryFrame(cap)) {
		frames++;
		float mse = 0.0;
		for (int i = 0; i < frameA->width; i++) {
			for (int j = 0; j < frameA->height; j++) {
				float redA = ((float*)(frameA->imageData + frameA->widthStep*j))[i*3];
				float greenA = ((float*)(frameA->imageData + frameA->widthStep*j))[i*3+1];
				float blueA = ((float*)(frameA->imageData + frameA->widthStep*j))[i*3+2];

				float redB = ((float*)(frameB->imageData + frameB->widthStep*j))[i*3];
				float greenB = ((float*)(frameB->imageData + frameB->widthStep*j))[i*3+1];
				float blueB = ((float*)(frameB->imageData + frameB->widthStep*j))[i*3+2];
				if (redA == 0 && greenA == 0 && blueA == 0) continue;
				if (redB == 0 && greenB == 0 && blueB == 0) continue;
				mse += (redA-redB + greenA-greenB + blueA-blueB)*(redA-redB + greenA-greenB + blueA-blueB);
			}
		}
		mse = mse/(frameA->width*frameA->height);
		PNSR += 10*log10((255+255+255)*(255+255+255))/mse;
        	cvCopy(frameB, frameA);
	}
	
	return PNSR/(double)frames;
	
}

IplImage* convertToGrayscale(IplImage *src) {
    IplImage *dst = cvCreateImage(cvSize(src->width, src->height), 8, 1);
    cvCvtColor(src, dst, CV_RGB2GRAY);
    
    return dst;
}

// usage: matchframes <video_filename> <num_frames_to_loop_through>
int main( int argc, char** argv )
{   
    argc--;
    argv++;
    char* input_filename = argv[0];
    char* output_filename = argv[1];
    int N = atoi(argv[2]);
    
    // create OpenCV storage
    CvMemStorage* storage = cvCreateMemStorage(0);
    
    IplImage* frameA = 0;
    IplImage* frameB = 0;
    
    CvCapture* cap = cvCaptureFromAVI(input_filename);

    // initialize data structures for SURF extraction
    CvSeq *frameKeypointsA = 0, *frameDescriptorsA = 0;
    CvSeq *frameKeypointsB = 0, *frameDescriptorsB = 0;
    CvSURFParams params = cvSURFParams(500, 1);

    double frames = cvGetCaptureProperty(cap, CV_CAP_PROP_FRAME_COUNT);
    cout << "# Frames: " << frames << endl;
    double frame_width = cvGetCaptureProperty(cap, CV_CAP_PROP_FRAME_WIDTH);
    double frame_height = cvGetCaptureProperty(cap, CV_CAP_PROP_FRAME_HEIGHT);
    cout << "Resolution: " << frame_width << " x " << frame_height << endl;
    double fps = cvGetCaptureProperty(cap, CV_CAP_PROP_FPS);
    cout << "FPS: " << fps << endl;

    CvSize frame_size = {frame_width, frame_height};

    CvVideoWriter* video_writer = cvCreateVideoWriter(output_filename, CV_FOURCC('F', 'M', 'P', '4'), fps, frame_size);

    int n = 0;

    frameB = cvQueryFrame(cap);
    frameA = cvCloneImage(frameB);

    double origin_x = frame_width/2;
    double origin_y = frame_height/2;

    CvMat *M = cvCreateMat(2, 3, CV_32F);	

    cvWriteFrame(video_writer, frameA);

	
   CvPoint2D32f srcTri[3];
   CvPoint2D32f dstTri[3];

    srcTri[0] = Point2f(0, 0);
    srcTri[1] = Point2f(100, 0);
    srcTri[2] = Point2f(0, 100);

    dstTri[0] = srcTri[0];
    dstTri[1] = srcTri[1];
    dstTri[2] = srcTri[2];

    cvQueryFrame(cap);

    while (n < N && (frameB = cvQueryFrame(cap))) {

        // convert frames to grayscale
        IplImage *gray_frameA = convertToGrayscale(frameA);
        IplImage *gray_frameB = convertToGrayscale(frameB);       

        // get SURF features and keypoints
        cvExtractSURF( gray_frameA, 0, &frameKeypointsA, &frameDescriptorsA, storage, params );
        cvExtractSURF( gray_frameB, 0, &frameKeypointsB, &frameDescriptorsB, storage, params );
        
        // validating keypoint extraction
        if (n == n) {
            showKeypoints(gray_frameB, frameKeypointsB);
	}
        
        // where the matching point pairs will be stored
        vector<int> ptpairs;
        
        // find matches based on Euclidean distance b/w descriptors w 1st/2nd NN threshold ratio
        flannFindPairs(frameKeypointsA, frameKeypointsB, ptpairs);
        
        double norms[ptpairs.size()/2];
        double mean = 0;
	for (int i=0; i < ptpairs.size(); i+=2) {
	    CvSURFPoint* pt_a = (CvSURFPoint*)cvGetSeqElem(frameKeypointsA, ptpairs[i]);
	    CvSURFPoint* pt_b = (CvSURFPoint*)cvGetSeqElem(frameKeypointsB, ptpairs[i+1]);
	    double delta_x = pt_b->pt.x-pt_a->pt.x;
            double delta_y = pt_b->pt.y-pt_a->pt.y;
	    double norm = sqrt(delta_x*delta_x+delta_y*delta_y);
	    norms[i/2] = norm;
	    mean += norm;
	}
	mean = 2*mean/ptpairs.size();
	double stddev = 0;
	for (int i=0; i < ptpairs.size()/2; i++) {
	    stddev+=(norms[i]-mean)*(norms[i]-mean);
        }
	stddev = sqrt(2*stddev/ptpairs.size());

	vector<int> screened_ptpairs;
	for (int i=0; i < ptpairs.size()/2; i++) {
	    if (norms[i]-mean < 2*stddev) {
		screened_ptpairs.push_back(ptpairs[2*i]);
		screened_ptpairs.push_back(ptpairs[2*i+1]);
            }
	}
	ptpairs = screened_ptpairs;

        // look up corresponding match pair keypoints by match indices and find differences
        // to compute local displacement vectors

	double distance [ptpairs.size()/2];
	double angle [ptpairs.size()/2];
	double lambdacostheta;
	double lambdasintheta;
	double Tx;
	double Ty;
	

        best_transform(frameKeypointsA, frameKeypointsB, origin_x, origin_y, ptpairs, distance, angle, &lambdacostheta, &lambdasintheta, &Tx, &Ty);


	printf("num_points: %d\n", ptpairs.size());
	printf("lambdacostheta: %f\n", lambdacostheta);
	printf("lambdasintheta: %f\n", lambdasintheta);
	printf("Tx: %f\n", Tx);
	printf("Ty: %f\n", Ty);

	vector<int> best_indices = fuzzy(distance, angle, ptpairs.size()/2);	
	screened_ptpairs.clear();
	for (int i=0; i < best_indices.size(); i++) {
		screened_ptpairs.push_back(ptpairs[2*best_indices[i]]);
		screened_ptpairs.push_back(ptpairs[2*best_indices[i] + 1]);
	}

	double screened_distance[screened_ptpairs.size()/2];
	double screened_angle[screened_ptpairs.size()/2];

	showArrows(gray_frameA, frameKeypointsA, frameKeypointsB, screened_ptpairs);

	best_transform(frameKeypointsA, frameKeypointsB, origin_x, origin_y, screened_ptpairs, screened_distance, screened_angle, &lambdacostheta, &lambdasintheta, &Tx, &Ty);

	printf("screened num_points: %d\n", screened_ptpairs.size());
	printf("screened lambdacostheta: %f\n", lambdacostheta);
	printf("screened lambdasintheta: %f\n", lambdasintheta);
	printf("screened Tx: %f\n", Tx);
	printf("screened Ty: %f\n", Ty);
	
        n++;

	for (int i=0; i<3; i++) {
	  double x_new = (dstTri[i].x-origin_x)*lambdacostheta-(dstTri[i].y-origin_y)*lambdasintheta+Tx+origin_x;
	  double y_new = (dstTri[i].x-origin_x)*lambdasintheta+(dstTri[i].y-origin_y)*lambdacostheta+Ty+origin_y;
          dstTri[i].x = x_new;
	  dstTri[i].y = y_new;
	}	

	/*
	for (int i=0; i<3; i++) {
	  dstTri[i].x = (srcTri[i].x-origin_x)*lambda*cos(theta)-(srcTri[i].y-origin_y)*lambda*sin(theta)+Tx+origin_x;
	  dstTri[i].y = (srcTri[i].x-origin_x)*lambda*sin(theta)+(srcTri[i].y-origin_y)*lambda*cos(theta)+Ty+origin_y;
	}
	*/

	cvGetAffineTransform(dstTri, srcTri, M);	
	cvWarpAffine(frameB, frameA, M);
	cvWriteFrame(video_writer, frameA);
        cvCopy(frameB, frameA);
    }

    cvReleaseMat(&M);
    cvReleaseCapture(&cap);
    cvReleaseVideoWriter(&video_writer);

    cout << endl;

   double initial_error = returnITF(input_filename);
   cout << "initial error: " << initial_error << endl;

   double output_error = returnITF(output_filename);
   cout << "final error: " << output_error << endl;
    
    return 0;
}
