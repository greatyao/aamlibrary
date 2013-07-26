/****************************************************************************
*						AAMLibrary
*			http://code.google.com/p/aam-library
* Copyright (c) 2008-2009 by GreatYao, all rights reserved.
****************************************************************************/

#ifndef AAM_CAM_H
#define AAM_CAM_H

#include "AAM_TDM.h"
#include "AAM_PDM.h"

class AAM_Basic;

//combined appearance model
class AAM_CAM  
{
	friend class AAM_Basic;
public:
	AAM_CAM();
	~AAM_CAM();

	// Build combined appearance model
	void Train(const file_lists& pts_files, const file_lists& img_files, 
		double scale = 1.0, double shape_percentage = 0.975, 
		double texture_percentage = 0.975, double appearance_percentage = 0.975);

	// Get dimension of combined appearance vector
	inline const int nParameters()const { return __AppearanceEigenVectors->cols;}

	// Get number of modes of combined appearance variation
	inline const int nModes()const { return __AppearanceEigenVectors->rows;}

	// Get variance of i-th mode of combined appearance variation
	inline double Var(int i)const { return cvmGet(__AppearanceEigenValues, 0, i); }

	// Get mean combined appearance
	inline const CvMat* GetMean()const{ return __MeanAppearance;	}

	// Get combined appearance eigen-vectors of PCA (appearance modes)
	inline const CvMat* GetBases()const{ return __AppearanceEigenVectors;	}

	// Show Model Variation according to various of parameters
	void ShowVariation();
	// function used in ShowVariation
	friend void ontrackcam(int pos);

	// Draw the image according the searching result
	void DrawAppearance(IplImage* image, const AAM_Shape& Shape, CvMat* Texture);

	// Calculate shape according to appearance parameters
	void CalcLocalShape(CvMat* s, const CvMat* c);
	void CalcGlobalShape(CvMat* s, const CvMat* pose);
	inline void CalcShape(CvMat* s, const CvMat* c_q)
	{	CvMat c; cvGetCols(c_q, &c, 4, 4+nModes()); CalcLocalShape(s, &c);
		CvMat q; cvGetCols(c_q, &q, 0, 4); CalcGlobalShape(s, &q);	}
	inline void CalcShape(CvMat* s, const CvMat* c, const CvMat* pose)
	{ CalcLocalShape(s, c); CalcGlobalShape(s, pose);	}
	
	
	// Calculate texture according to appearance parameters
	void CalcTexture(CvMat* t, const CvMat* c);
	
	//Calculate combined appearance parameters from shape and texture params.
	void CalcParams(CvMat* c, const CvMat* p, const CvMat* lamda);

	// Limit appearance parameters.
    void Clamp(CvMat* c, double s_d = 3.0);

	// Read data from stream 
	void Read(std::ifstream& is);

	// Write data to stream
	void Write(std::ofstream& os);

private:
	// Do PCA of appearance datas
	void DoPCA(const CvMat* AllAppearances, double percentage);

	// Convert shape and texture instance to appearance parameters
	void ShapeTexture2Combined(const CvMat* Shape, const CvMat* Texture, 
		CvMat* Appearance);

private:
	AAM_PDM		__shape;		/*shape distribution model*/
	AAM_TDM		__texture;		/*shape distribution model*/
	AAM_PAW		__paw;			/*piecewise affine warp*/
	double      __WeightsS2T;   /*ratio between shape and texture model*/
	
	CvMat* __MeanAppearance;
	CvMat* __AppearanceEigenValues;
	CvMat* __AppearanceEigenVectors;
	
	CvMat* __Qs;
	CvMat* __Qg;
	CvMat* __MeanS;
	CvMat* __MeanG;

	CvMat* __a;

private:
	//these cached variables are used for speed up
	CvMat*			__Points;
	CvMemStorage*	__Storage;
	CvMat*			__pq;
};

#endif // !AAM_CAM_H
