/****************************************************************************
*						AAMLibrary
*			http://code.google.com/p/aam-library
* Copyright (c) 2008-2009 by GreatYao, all rights reserved.
****************************************************************************/

#include "AAM_IC.h"
#include "AAM_Basic.h"
#include "AAM_MovieAVI.h"
#include "VJfacedetect.h"

using namespace std;

static void usage()
{
	printf("Usage: fit model_file cascade_file (image/video)_file\n");
	exit(0);
}

int main(int argc, char** argv)
{
	if(argc != 4)	usage();
	
	AAM_Pyramid model;
	model.ReadModel(argv[1]);
	VJfacedetect facedet;
	facedet.LoadCascade(argv[2]);
	char filename[100];
	strcpy(filename, argv[3]);	

	if(strstr(filename, ".avi"))
	{
		AAM_MovieAVI aviIn;
		AAM_Shape Shape;
		IplImage* image2 = 0;
		bool flag = false;
		
		aviIn.Open(filename);
		cvNamedWindow("Video",1);
		cvNamedWindow("AAMFitting",1);
		
		for(int j = 0; j < aviIn.FrameCount(); j ++)
		{
			printf("Tracking frame %04i: ", j);

			IplImage* image = aviIn.ReadFrame(j);

			if(j == 0 || flag == false) 
			{
				flag = model.InitShapeFromDetBox(Shape, facedet, image);
				if(flag == false) goto show;
			}
	
			flag = model.Fit(image, Shape, 30, false);
			if(image2 == 0) image2 = cvCreateImage(cvGetSize(image), image->depth, image->nChannels);
			cvZero(image2);
			model.Draw(image2, Shape, 2);
			cvShowImage("AAMFitting", image2);
show:
			cvShowImage("Video", image);
			cvWaitKey(1);
		}
		cvReleaseImage(&image2);
	}
	
	else
	{
		IplImage* image = cvLoadImage(filename, -1);
		AAM_Shape Shape;
		bool flag = flag = model.InitShapeFromDetBox(Shape, facedet, image);
		if(flag == false) {
			fprintf(stderr, "The image doesn't contain any faces\n");
			exit(0);
		}
		model.Fit(image, Shape, 30, true);
		model.Draw(image, Shape, 2);
		
		cvNamedWindow("Fitting");
		cvShowImage("Fitting", image);
		cvWaitKey(0);
		
		cvReleaseImage(&image);	
	}

	return 0;
}
