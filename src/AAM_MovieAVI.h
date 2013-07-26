/****************************************************************************
*						AAMLibrary
*			http://code.google.com/p/aam-library
* Copyright (c) 2008-2009 by GreatYao, all rights reserved.
****************************************************************************/

#ifndef AAM_MOVIEAVI_H
#define AAM_MOVIEAVI_H

#include "cv.h"
#include "highgui.h"

class AAM_MovieAVI
{
public:
	AAM_MovieAVI(): capimg(0), capture(0), image(0){	}
	~AAM_MovieAVI(){Close();}

	// Open a AVI file
	void Open(const char* videofile);
	
	// Close it
	void Close();

	// Get concrete frame of the video
	// Notice: for speed up you have no need to release the returned image
	IplImage* ReadFrame(int frame_no = -1);
	
	// frame count of this video
	const int FrameCount()const
	{return (int)cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_COUNT);}

private:
	IplImage* capimg;//captured from video
	IplImage *image;
	CvCapture* capture;

};

#endif // !
