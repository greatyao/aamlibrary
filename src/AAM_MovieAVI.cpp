/****************************************************************************
*						AAMLibrary
*			http://code.google.com/p/aam-library
* Copyright (c) 2008-2009 by GreatYao, all rights reserved.
****************************************************************************/

#include <iostream>
#include "AAM_MovieAVI.h"

//============================================================================
void AAM_MovieAVI::Open(const char* videofile)
{
	capture = cvCaptureFromAVI(videofile);
	if(!capture)
	{
		fprintf(stderr, "could not open video file %s!\n", videofile);
		exit(0);
	}
	cvSetCaptureProperty(capture, CV_CAP_PROP_POS_FRAMES, 0);
	capimg = cvQueryFrame( capture );
	image = cvCreateImage(cvGetSize(capimg), capimg->depth, capimg->nChannels);
}

//============================================================================
void AAM_MovieAVI::Close()
{
	cvReleaseCapture(&capture);
	capture = 0;
	cvReleaseImage(&image);
	image = 0;
}

//============================================================================
IplImage* AAM_MovieAVI:: ReadFrame(int frame_no )
{
	cvSetCaptureProperty(capture, CV_CAP_PROP_POS_FRAMES, frame_no);
	capimg = cvQueryFrame( capture );

	if(capimg->origin == 0)
		cvCopy(capimg, image);
	else
		cvFlip(capimg, image);

	return image;
}