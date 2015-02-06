/****************************************************************************
*						AAMLibrary
*			http://code.google.com/p/aam-library
* Copyright (c) 2008-2009 by GreatYao, all rights reserved.
****************************************************************************/

#ifndef AAM_UTIL_H
#define AAM_UTIL_H

#include <vector>
#include <string>
#include <iostream>
#include <fstream>

#include "cv.h"
#include "highgui.h"

#include "AAM_Shape.h"
#include "VJfacedetect.h"

#ifndef byte
#define byte unsigned char
#endif

#define TYPE_AAM_BASIC	0
#define TYPE_AAM_IC		1

#define gettime cvGetTickCount() / (cvGetTickFrequency()*1000.)

#ifdef ANDROID
    #include <android/log.h>
    #define LOG_TAG "AAMLIBRARY"
    #define LOGD(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__))
    #define LOGI(...) ((void)__android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__))
    #define LOGW(...) ((void)__android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__))
#else
	#define LOGD(...) fprintf(stdout,  __VA_ARGS__)
	#define LOGI(...) fprintf(stdout,  __VA_ARGS__)
	#define LOGW(...) fprintf(stderr,  __VA_ARGS__)
#endif

typedef std::vector<std::string> file_lists;

void ReadCvMat(std::istream &is, CvMat* mat);

void WriteCvMat(std::ostream &os, const CvMat* mat);


class AAM_PAW;

class AAM_Common
{
public:
	static file_lists ScanNSortDirectory(const std::string &path, 
		const std::string &extension);

	// Is the current shape within the image boundary?
	static void CheckShape(CvMat* s, int w, int h);

	static void DrawPoints(IplImage* image, const AAM_Shape& Shape);

	static void DrawTriangles(IplImage* image, const AAM_Shape& Shape, 
		const std::vector<std::vector<int> >&tris);
	
	static void DrawAppearance(IplImage*image, const AAM_Shape& Shape,
		const CvMat* t, const AAM_PAW& paw, const AAM_PAW& refpaw);

	static int MkDir(const char* dirname);

};

//virtual class for Active Appearance Model
class AAM
{
public:
	AAM();
	virtual ~AAM() = 0;

	virtual const int GetType()const = 0;

	// Build aam model
	virtual void Build(const file_lists& pts_files, 
		const file_lists& img_files, double scale = 1.0) = 0;
	
	// Fit the image using aam 
	virtual bool Fit(const IplImage* image, AAM_Shape& Shape, 
		int max_iter = 30, bool showprocess = false) = 0;

	// Set search parameters zero
	virtual void SetAllParamsZero() = 0;

	// Init search parameters 
	virtual void InitParams(const IplImage* image) = 0;

	// Draw the image according search result
	virtual void Draw(IplImage* image, const AAM_Shape& Shape, int type) = 0;

	// Read data from stream 
	virtual void Read(std::ifstream& is) = 0;

	// Write data to stream
	virtual void Write(std::ofstream& os) = 0;

	// Get Mean Shape of model
	virtual const AAM_Shape GetMeanShape()const = 0;
	virtual const AAM_Shape GetReferenceShape()const = 0;
};

class AAM_Pyramid
{
public:
	AAM_Pyramid();
	~AAM_Pyramid();
	
	// Build Multi-Resolution Active Appearance Model
	void Build(const file_lists& pts_files, const file_lists& img_files,
		int type = TYPE_AAM_IC, int level = 1);

	// Doing image alignment
	bool Fit(const IplImage* image, AAM_Shape& Shape, 
		int max_iter = 30, bool showprocess = false);

	// Build mapping relation between detect box and shape
	void BuildDetectMapping(const file_lists& pts_files, 
		const file_lists& img_files, VJfacedetect& facedetect,
		double refWidth = 100);

	// Init shape from the mapping  
	bool InitShapeFromDetBox(AAM_Shape& Shape, VJfacedetect& facedetect, 
		const IplImage* image);
		
	void InitShapeFromDetBox(AAM_Shape &Shape, const AAM_Shape& DetShapeBox);

	// Write aam to file
	bool WriteModel(const std::string& filename);

	// Read aam from file
	bool ReadModel(const std::string& filename);

	// Draw the image according search result
	void Draw(IplImage* image, const AAM_Shape& Shape, int type);

	// Get Mean Shape of model
	const AAM_Shape GetMeanShape()const;
	
private:
	std::vector<AAM*>	__model;
	AAM_Shape			__VJDetectShape;
	double				__referenceWidth;

};

#endif // AAM_UTIL_H
