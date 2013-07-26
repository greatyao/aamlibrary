/****************************************************************************
*						AAMLibrary
*			http://code.google.com/p/aam-library
* Copyright (c) 2008-2009 by GreatYao, all rights reserved.
****************************************************************************/

#include "VJfacedetect.h"

using namespace std;

//============================================================================
VJfacedetect::VJfacedetect()
{
	__cascade = 0;
	__storage = 0;
}

//============================================================================
VJfacedetect::~VJfacedetect()
{
	cvReleaseMemStorage(&__storage);
	cvReleaseHaarClassifierCascade(&__cascade);
}

//============================================================================
void VJfacedetect::LoadCascade(const char* cascade_name /* = "haarcascade_frontalface_alt2.xml" */)
{
	__cascade = (CvHaarClassifierCascade*)cvLoad(cascade_name, 0, 0, 0);
	if(__cascade == 0)
	{
		printf("ERROR(%s, %d): Can't load cascade file!\n", __FILE__, __LINE__);
		exit(0);
	}	
	__storage = cvCreateMemStorage(0);
}

//============================================================================
bool VJfacedetect::DetectFace(std::vector<AAM_Shape> &Shape, const IplImage* image)
{
	IplImage* small_image = cvCreateImage
		(cvSize(image->width/2, image->height/2), image->depth, image->nChannels);
	cvPyrDown(image, small_image, CV_GAUSSIAN_5x5);
		
	CvSeq* pFaces = cvHaarDetectObjects(small_image, __cascade, __storage,
		1.1, 3, CV_HAAR_DO_CANNY_PRUNING);

	cvReleaseImage(&small_image);
	
	if(0 == pFaces->total)//can't find a face
		return false;

	Shape.resize(pFaces->total);
	for (int i = 0; i < pFaces->total; i++)
    {
		Shape[i].resize(2);
		CvRect* r = (CvRect*)cvGetSeqElem(pFaces, i);
		
		CvPoint pt1, pt2;
		pt1.x = r->x * 2;
		pt2.x = (r->x + r->width) * 2;
		pt1.y = r->y * 2;
		pt2.y = (r->y + r->height) * 2;
	
		Shape[i][0].x  = r->x*2.0;
		Shape[i][0].y  = r->y*2.0;
		Shape[i][1].x  = Shape[i][0].x + 2.0*r->width;
		Shape[i][1].y  = Shape[i][0].y + 2.0*r->height;
    }
	return true;
}
