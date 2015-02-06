#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "AAM_Util.h"
#include "AAM_IC.h"
#include "AAM_Basic.h"
#include "VJfacedetect.h"

#include <string>
#include <vector>

#include <jni.h>

using namespace std;
using namespace cv;

#define BEGINT()	double t = (double)cvGetTickCount();
#define ENDT(exp)	t = ((double)cvGetTickCount() -  t )/  (cvGetTickFrequency()*1000.);	\
					LOGD(exp " time cost: %.2f millisec\n", t);

AAM_Pyramid model;
VJfacedetect facedet;

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT jboolean JNICALL Java_me_greatyao_AAMFit_nativeReadModel
(JNIEnv * jenv, jclass, jstring jFileName)
{
    LOGD("nativeReadModel enter");
    const char* filename = jenv->GetStringUTFChars(jFileName, NULL);
    jboolean result = false;

    try
    {
	if(model.ReadModel(filename) == true)
		result = true;

    }
    catch (...)
    {
        LOGD("nativeReadModel caught unknown exception");
        jclass je = jenv->FindClass("java/lang/Exception");
        jenv->ThrowNew(je, "Unknown exception in JNI code");
    }

    LOGD("nativeReadModel %s exit %d", filename, result);
    return result;
}

JNIEXPORT jboolean JNICALL Java_me_greatyao_AAMFit_nativeInitCascadeDetector
(JNIEnv * jenv, jclass, jstring jFileName)
{
    const char* cascade_name = jenv->GetStringUTFChars(jFileName, NULL);
    LOGD("nativeInitCascadeDetector %s enter", cascade_name);

    if(facedet.LoadCascade(cascade_name) == false)
		return false;

    LOGD("nativeInitCascadeDetector exit");
    return true;
}

inline void shape_to_Mat(AAM_Shape shapes[], int nShape, Mat& mat)
{
	mat = Mat(nShape, shapes[0].NPoints()*2, CV_64FC1); 

	for(int i = 0; i < nShape; i++)
	{
		double *pt = mat.ptr<double>(i);  
		for(int j = 0; j < mat.cols/2; j++)
		{
			pt[2*j] = shapes[i][j].x;
			pt[2*j+1] = shapes[i][j].y;		
		}
	}
}

inline void Mat_to_shape(AAM_Shape shapes[], int nShape, Mat& mat)
{
	for(int i = 0; i < nShape; i++)
	{
		double *pt = mat.ptr<double>(i);  
		shapes[i].resize(mat.cols/2);
		for(int j = 0; j < mat.cols/2; j++)
		{
			shapes[i][j].x = pt[2*j];
			shapes[i][j].y = pt[2*j+1];
		}
	}
}

JNIEXPORT void JNICALL Java_me_greatyao_AAMFit_nativeInitShape(JNIEnv * jenv, jclass, jlong faces)
{
	Mat faces1 = *((Mat*)faces);
	int nFaces = faces1.rows;
	AAM_Shape* detshapes = new AAM_Shape[nFaces];
	AAM_Shape* shapes = new AAM_Shape[nFaces];

	Mat_to_shape(detshapes, nFaces, faces1);

	for(int i = 0; i < nFaces; i++)
	{
		if(detshapes[i].NPoints() != 2) continue;
		model.InitShapeFromDetBox(shapes[i], detshapes[i]);
	}

	shape_to_Mat(shapes, nFaces, *((Mat*)faces));

	delete []detshapes;
	delete []shapes;
}


JNIEXPORT jboolean JNICALL Java_me_greatyao_AAMFit_nativeDetectOne
(JNIEnv * jenv, jclass, jlong imageGray, jlong faces)
{
	IplImage image = *(Mat*)imageGray;
	AAM_Shape shape, detshape;
	
	BEGINT();
	
	std::vector<AAM_Shape> detshapes;
	bool flag = facedet.DetectFace(detshapes, &image);
	if(flag == false)	{
		ENDT("CascadeDetector CANNOT detect any face");
		return false;
	}
	detshape = detshapes[0];

	model.InitShapeFromDetBox(shape, detshape);

	shape_to_Mat(&shape, 1, *((Mat*)faces));
	
	ENDT("CascadeDetector detects one face");

	return true;
}

JNIEXPORT jboolean JNICALL Java_me_greatyao_AAMFit_nativeFitting
(JNIEnv * jenv, jclass, jlong imageGray, jlong shapes0, jlong n_iteration)
{
	IplImage image = *(Mat*)imageGray;
	Mat shapes1 = *(Mat*)shapes0;	
	bool flag = false;
	assert(shapes1.rows == 1);
	
	{
		AAM_Shape shape;
	
		BEGINT();

		Mat_to_shape(&shape, 1, shapes1);

		flag = model.Fit(&image, shape, (int)n_iteration, false);

		shape_to_Mat(&shape, 1, *((Mat*)shapes0));

		ENDT("nativeVideoFitting");
	}

	return flag;
}

JNIEXPORT void JNICALL Java_me_greatyao_AAMFit_nativeDrawImage
(JNIEnv * jenv, jclass, jlong imageColor, jlong shapes0, jlong type)
{
	IplImage image = *(Mat*)imageColor;
	Mat shapes1 = *(Mat*)shapes0;	
	assert(shapes1.rows == 1);
	
	{
		AAM_Shape shape;
	
		BEGINT();

		Mat_to_shape(&shape, 1, shapes1);

		model.Draw(&image, shape, type);

		ENDT("nativeDrawImage");
	}
}


#ifdef __cplusplus
}
#endif

