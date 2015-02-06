/****************************************************************************
*						AAMLibrary
*			http://code.google.com/p/aam-library
* Copyright (c) 2008-2009 by GreatYao, all rights reserved.
****************************************************************************/

#include "AAM_IC.h"


//============================================================================
AAM_IC::AAM_IC()
{
	__Points = 0;
	__Storage = 0;

	__update_s0 = 0;
	__warp_t = 0;
	__error_t = 0;
	__search_pq = 0;
	__delta_pq = 0;
	__current_s = 0;
	__update_s = 0;
}

//============================================================================
AAM_IC::~AAM_IC()
{
	cvReleaseMat(&__Points);
	cvReleaseMemStorage(&__Storage);

	cvReleaseMat(&__update_s0);
	cvReleaseMat(&__warp_t);
	cvReleaseMat(&__error_t);
	cvReleaseMat(&__search_pq);
	cvReleaseMat(&__delta_pq);
	cvReleaseMat(&__current_s);
	cvReleaseMat(&__update_s);
}

//============================================================================
CvMat* AAM_IC::CalcGradIdx()
{
	CvMat* pos= cvCreateMat(__paw.nPix(), 4, CV_32SC1);

	int i = 0;
	int width = __paw.Width(), height = __paw.Height();
	for(int y = 0; y < height; y++)
	{
		for(int x = 0; x < width; x++)
		{
			if(__paw.Rect(y, x) >= 0)
			{
				int *ppos = (int*)(pos->data.ptr + 	pos->step*i);
				ppos[0] = (x-1<0)		?-1:__paw.Rect(y, x-1);  // left
				ppos[1] = (x+1>=width)	?-1:__paw.Rect(y, x+1);  // right
				ppos[2] = (y-1<0)		?-1:__paw.Rect(y-1, x);  // top
				ppos[3] = (y+1>=height)	?-1:__paw.Rect(y+1, x);  // bottom
				i++;
			}
		}
	}
	
	return pos;
}

//============================================================================
void AAM_IC::CalcTexGrad(const CvMat* texture, CvMat* dTx, CvMat* dTy)
{
	double* _x = dTx->data.db;
	double* _y = dTy->data.db;
	double* t = texture->data.db;
	CvMat *idx = CalcGradIdx();

	for(int i = 0; i < __paw.nPix(); i++)
	{
		int *fastp = (int*)(idx->data.ptr + idx->step*i);
				
		// x direction
		if(fastp[0] >= 0 && fastp[1] >= 0)
		{
			_x[3*i+0] = (t[3*fastp[1]+0] - t[3*fastp[0]+0])/2;
			_x[3*i+1] = (t[3*fastp[1]+1] - t[3*fastp[0]+1])/2;
			_x[3*i+2] = (t[3*fastp[1]+2] - t[3*fastp[0]+2])/2;
		}
		
		else if(fastp[0] >= 0 && fastp[1] < 0)
		{
			_x[3*i+0] = t[3*i+0] - t[3*fastp[0]+0];
			_x[3*i+1] = t[3*i+1] - t[3*fastp[0]+1];
			_x[3*i+2] = t[3*i+2] - t[3*fastp[0]+2];
		}
		
		else if(fastp[0] < 0 && fastp[1] >= 0)
		{
			_x[3*i+0] = t[3*fastp[1]+0] - t[3*i+0];
			_x[3*i+1] = t[3*fastp[1]+1] - t[3*i+1];
			_x[3*i+2] = t[3*fastp[1]+2] - t[3*i+2];
		}
		else
		{
			_x[3*i+0] = 0;
			_x[3*i+1] = 0;
			_x[3*i+2] = 0;
		}
		
		// y direction
		if(fastp[2] >= 0 && fastp[3] >= 0)
		{
			_y[3*i+0] = (t[3*fastp[3]+0] - t[3*fastp[2]+0])/2;
			_y[3*i+1] = (t[3*fastp[3]+1] - t[3*fastp[2]+1])/2;
			_y[3*i+2] = (t[3*fastp[3]+2] - t[3*fastp[2]+2])/2;
		}
		
		else if(fastp[2] >= 0 && fastp[3] < 0)
		{
			_y[3*i+0] = t[3*i+0] - t[3*fastp[2]+0];
			_y[3*i+1] = t[3*i+1] - t[3*fastp[2]+1];
			_y[3*i+2] = t[3*i+2] - t[3*fastp[2]+2];
		}
		
		else if(fastp[2] < 0 && fastp[3] >= 0)
		{
			_y[3*i+0] = t[3*fastp[3]+0] - t[3*i+0];
			_y[3*i+1] = t[3*fastp[3]+1] - t[3*i+1];
			_y[3*i+2] = t[3*fastp[3]+2] - t[3*i+2];
		}
		
		else
		{
			_y[3*i+0] = 0;
			_y[3*i+1] = 0;
			_y[3*i+2] = 0;
		}
	}
	cvReleaseMat(&idx);
}

//============================================================================
void AAM_IC::CalcWarpJacobian(CvMat* Jx, CvMat* Jy)
{
	int nPoints = __shape.nPoints();
	__sMean.Mat2Point(__shape.GetMean());
	__sStar1.resize(nPoints); __sStar2.resize(nPoints);
	__sStar3.resize(nPoints); __sStar4.resize(nPoints);
	for(int n = 0; n < nPoints; n++) // Equation (43)
	{
		__sStar1[n].x = __sMean[n].x;   __sStar1[n].y = __sMean[n].y;
		__sStar2[n].x = -__sMean[n].y;  __sStar2[n].y = __sMean[n].x;
		__sStar3[n].x = 1;			    __sStar3[n].y = 0;
		__sStar4[n].x = 0;				__sStar4[n].y = 1;
	}

	const CvMat* B = __shape.GetBases();
	const CvMat* mean = __shape.GetMean();
	cvZero(Jx); cvZero(Jy);
	for(int i = 0; i < __paw.nPix(); i++)
	{
		int tri_idx = __paw.PixTri(i);
		int v1 = __paw.Tri(tri_idx, 0);
		int v2 = __paw.Tri(tri_idx, 1);
		int v3 = __paw.Tri(tri_idx, 2);
		double *fastJx = (double*)(Jx->data.ptr + Jx->step*i);
		double *fastJy = (double*)(Jy->data.ptr + Jy->step*i);
		
		// Equation (50) dN_dq
		fastJx[0] = __paw.Alpha(i)*__sStar1[v1].x +
			__paw.Belta(i)*__sStar1[v2].x +  __paw.Gamma(i)*__sStar1[v3].x;
		fastJy[0] = __paw.Alpha(i)*__sStar1[v1].y +
			__paw.Belta(i)*__sStar1[v2].y +  __paw.Gamma(i)*__sStar1[v3].y;
		
		fastJx[1] = __paw.Alpha(i)*__sStar2[v1].x +
			__paw.Belta(i)*__sStar2[v2].x +  __paw.Gamma(i)*__sStar2[v3].x;
		fastJy[1] = __paw.Alpha(i)*__sStar2[v1].y +
			__paw.Belta(i)*__sStar2[v2].y +  __paw.Gamma(i)*__sStar2[v3].y;

		fastJx[2] = __paw.Alpha(i)*__sStar3[v1].x +
			__paw.Belta(i)*__sStar3[v2].x +  __paw.Gamma(i)*__sStar3[v3].x;
		fastJy[2] = __paw.Alpha(i)*__sStar3[v1].y +
			__paw.Belta(i)*__sStar3[v2].y +  __paw.Gamma(i)*__sStar3[v3].y;

		fastJx[3] = __paw.Alpha(i)*__sStar4[v1].x +
			__paw.Belta(i)*__sStar4[v2].x +  __paw.Gamma(i)*__sStar4[v3].x;
		fastJy[3] = __paw.Alpha(i)*__sStar4[v1].y +
			__paw.Belta(i)*__sStar4[v2].y +  __paw.Gamma(i)*__sStar4[v3].y;

		// Equation (51) dW_dp
		for(int j = 0; j < __shape.nModes(); j++)
		{
			fastJx[j+4] = __paw.Alpha(i)*cvmGet(B,j,2*v1) +
				__paw.Belta(i)*cvmGet(B,j,2*v2) + __paw.Gamma(i)*cvmGet(B,j,2*v3);			
			
			fastJy[j+4] = __paw.Alpha(i)*cvmGet(B,j,2*v1+1) +
				__paw.Belta(i)*cvmGet(B,j,2*v2+1) + __paw.Gamma(i)*cvmGet(B,j,2*v3+1);	
		}
	}
}

//============================================================================
void AAM_IC::CalcModifiedSD(CvMat* SD, const CvMat* dTx, const CvMat* dTy, 
							const CvMat* Jx, const CvMat* Jy)
{
	int i, j;
	
	//create steepest descent images
	double* _x = dTx->data.db;
	double* _y = dTy->data.db;
	double temp;
	for(i = 0; i < __shape.nModes()+4; i++)
	{
		for(j = 0; j < __paw.nPix(); j++)
		{
			temp = _x[3*j  ]*cvmGet(Jx,j,i) +_y[3*j  ]*cvmGet(Jy,j,i);
			cvmSet(SD,i,3*j,temp); 

			temp = _x[3*j+1]*cvmGet(Jx,j,i) +_y[3*j+1]*cvmGet(Jy,j,i);
			cvmSet(SD,i,3*j+1,temp); 

			temp = _x[3*j+2]*cvmGet(Jx,j,i) +_y[3*j+2]*cvmGet(Jy,j,i);
			cvmSet(SD,i,3*j+2,temp); 
		}
	}

	//project out appearance variation (and linear lighting parameters)
	const CvMat* B = __texture.GetBases();
	CvMat* V = cvCreateMat(4+__shape.nModes(), __texture.nModes(), CV_64FC1);
	CvMat SDMat, BMat;
	
	cvGEMM(SD, B, 1., NULL, 1., V, CV_GEMM_B_T);
	// Equation (63),(64)
	for(i = 0; i < __shape.nModes()+4; i++)
	{
		for(j = 0; j < __texture.nModes(); j++)
		{
			cvGetRow(SD, &SDMat, i);
			cvGetRow(B, &BMat, j);
			cvScaleAdd(&BMat, cvScalar(-cvmGet(V,i,j)), &SDMat, &SDMat);
		}
	}

	cvReleaseMat(&V);
}

//============================================================================
void AAM_IC::CalcHessian(CvMat* H, const CvMat* SD)
{
	CvMat* HH = cvCreateMat(H->rows, H->cols, CV_64FC1);
	cvMulTransposed(SD, HH, 0);// Equation (65)
	cvInvert(HH, H, CV_SVD);
	cvReleaseMat(&HH);
}

//============================================================================
void AAM_IC::Train(const file_lists& pts_files, 
				   const file_lists& img_files, 
				   double scale /* = 1.0 */, 
				   double shape_percentage /* = 0.975 */, 
				   double texture_percentage /* = 0.975 */)
{
	if(pts_files.size() != img_files.size())
	{
		LOGW("ERROE(%s, %d): #Shapes != #Images\n", __FILE__, __LINE__);
		exit(0);
	}

	LOGD("################################################\n");
	LOGD("Build Inverse Compositional Image Alignmennt Model...\n");

	std::vector<AAM_Shape> AllShapes;
	for(int ii = 0; ii < pts_files.size(); ii++)
	{
		AAM_Shape Shape;
		bool flag = Shape.ReadAnnotations(pts_files[ii]);
		if(!flag)
		{
			IplImage* image = cvLoadImage(img_files[ii].c_str(), -1);
			Shape.ScaleXY(image->width, image->height);
			cvReleaseImage(&image);
		}
		AllShapes.push_back(Shape);
	}

	//building shape and texture distribution model
	LOGD("Build point distribution model...\n");
	__shape.Train(AllShapes, scale, shape_percentage);
	
	LOGD("Build warp information of mean shape mesh...");
	__Points = cvCreateMat (1, __shape.nPoints(), CV_32FC2);
	__Storage = cvCreateMemStorage(0);

	double sp = 1.0;
	//if(__shape.GetMeanShape().GetWidth() > 150)
	//	sp = 150/__shape.GetMeanShape().GetWidth();

	__paw.Train(__shape.GetMeanShape()*sp, __Points, __Storage);
	LOGD("[%d by %d, triangles #%d, pixels #%d*3]\n",
		__paw.Width(), __paw.Height(), __paw.nTri(), __paw.nPix());

	LOGD("Build texture distribution model...\n");
	__texture.Train(pts_files, img_files, __paw, texture_percentage, true);

	//calculate gradient of texture
	LOGD("Calculating texture gradient...\n");
	CvMat* dTx = cvCreateMat(1, __texture.nPixels(), CV_64FC1);
	CvMat* dTy = cvCreateMat(1, __texture.nPixels(), CV_64FC1);
	CalcTexGrad(__texture.GetMean(), dTx, dTy);
	
	// save gradient image
	AAM_Common::MkDir("Modes");
	__paw.SaveWarpTextureToImage("Modes/dTx.jpg", dTx);
	__paw.SaveWarpTextureToImage("Modes/dTy.jpg", dTy);
	
	//calculate warp Jacobian at base shape
	LOGD("Calculating warp Jacobian...\n");
	CvMat* Jx = cvCreateMat(__paw.nPix(), __shape.nModes()+4, CV_64FC1);
	CvMat* Jy = cvCreateMat(__paw.nPix(), __shape.nModes()+4, CV_64FC1);
	CalcWarpJacobian(Jx,Jy);
	
	//calculate modified steepest descent image
	LOGD("Calculating steepest descent images...\n");
	CvMat* SD = cvCreateMat(__shape.nModes()+4, __texture.nPixels(), CV_64FC1);
	CalcModifiedSD(SD, dTx, dTy, Jx, Jy);

	//calculate inverse Hessian matrix
	LOGD("Calculating Hessian inverse matrix...\n");
	CvMat* H = cvCreateMat(__shape.nModes()+4, __shape.nModes()+4, CV_64FC1);
	CalcHessian(H, SD);

	//calculate update matrix (multiply inverse Hessian by modified steepest descent image)
	__G = cvCreateMat(__shape.nModes()+4, __texture.nPixels(), CV_64FC1);
	cvMatMul(H, SD, __G);

	//release
	cvReleaseMat(&Jx);
	cvReleaseMat(&Jy);
	cvReleaseMat(&dTx);
	cvReleaseMat(&dTy);
	cvReleaseMat(&SD);
	cvReleaseMat(&H);

	//alocate memory for on-line fitting stuff
	__update_s0 = cvCreateMat(1, __shape.nPoints()*2, CV_64FC1);
	__inv_pq = cvCreateMat(1, __shape.nModes()+4, CV_64FC1);
	__warp_t = cvCreateMat(1, __texture.nPixels(), CV_64FC1);
	__error_t = cvCreateMat(1, __texture.nPixels(), CV_64FC1);
	__search_pq = cvCreateMat(1, __shape.nModes()+4, CV_64FC1);
	__delta_pq = cvCreateMat(1, __shape.nModes()+4, CV_64FC1);
	__current_s = cvCreateMat(1, __shape.nPoints()*2, CV_64FC1);
	__update_s = cvCreateMat(1, __shape.nPoints()*2, CV_64FC1);
	__lamda  = cvCreateMat(1, __texture.nModes(), CV_64FC1);

	LOGD("################################################\n\n");
}

//============================================================================
bool AAM_IC::Fit(const IplImage* image, 		AAM_Shape& Shape, 
				int max_iter /* = 30 */, 	bool showprocess /* = false */)
{
	//initialize some stuff
	double t = gettime;
	const CvMat* A0 = __texture.GetMean();
	CvMat p; cvGetCols(__search_pq, &p, 4, 4+__shape.nModes());
	Shape.Point2Mat(__current_s);
	SetAllParamsZero();
	__shape.CalcParams(__current_s, __search_pq);
	IplImage* Drawimg = 0;
	double err1, err2;
	
	for(int iter = 0; iter < max_iter; iter++)
	{
		if(showprocess)
		{	
			if(Drawimg == 0)	Drawimg = cvCloneImage(image);	
			else cvCopy(image, Drawimg);
			Shape.Mat2Point(__current_s);
			Draw(Drawimg, Shape, 2);
			AAM_Common::MkDir("result");
			char filename[100];
			sprintf(filename, "result/Iter-%02d.jpg", iter);
			cvSaveImage(filename, Drawimg);
			
		}
		
		//check the current shape
		AAM_Common::CheckShape(__current_s, image->width, image->height);
		
		//warp image to mesh shape mesh
		__paw.CalcWarpTexture(__current_s, image, __warp_t);
		AAM_TDM::NormalizeTexture(A0, __warp_t);
		cvSub(__warp_t, A0, __error_t);
		
		 //calculate updates (and scale to account for linear lighting gain)
		cvGEMM(__error_t, __G, 1, NULL, 1, __delta_pq, CV_GEMM_B_T);
		
		//check for parameter convergence
		if((err1=cvNorm(__delta_pq)) < 1e-6)	break;

		//apply inverse compositional algorithm to update parameters
		InverseCompose(__delta_pq, __current_s, __update_s);
		
		//smooth shape
		cvAddWeighted(__current_s, 0.4, __update_s, 0.6, 0, __update_s);
		//update parameters
		__shape.CalcParams(__update_s, __search_pq);
		//calculate constrained new shape
		__shape.CalcShape(__search_pq, __update_s);
		
		//check for shape convergence
		if((err2=cvNorm(__current_s, __update_s, CV_L2)) < 0.001)	break;
		else cvCopy(__update_s, __current_s);	
	}
	
	Shape.Mat2Point(__current_s);
		
	t = gettime-t;
	LOGI("AAM-IC Fitting: time cost=%.3f millisec, measure=(%.2f, %.2f)\n", t, err1, err2);
	
	cvReleaseImage(&Drawimg);
	
	if(err1 >= 15 || err2 >= 3.75) return false;
	return true;
}

//============================================================================
void AAM_IC::SetAllParamsZero()
{
	cvZero(__warp_t);
	cvZero(__error_t);
	cvZero(__search_pq);
	cvZero(__delta_pq);
	cvZero(__lamda);
}

//============================================================================
void AAM_IC::InverseCompose(const CvMat* dpq, const CvMat* s, CvMat* NewS)
{
	// Firstly: Estimate the corresponding changes to the base mesh
	cvConvertScale(dpq, __inv_pq, -1);
	__shape.CalcShape(__inv_pq, __update_s0);	// __update_s0 = N.W(s0, -delta_p, -delta_q)

	//Secondly: Composing the Incremental Warp with the Current Warp Estimate.
	double *S0 = __update_s0->data.db;
	double *S = s->data.db;
	double *SEst = NewS->data.db;
	double x, y, xw, yw;
	int k, tri_idx;
	int v1, v2, v3;
	const std::vector<std::vector<int> >& tri = __paw.__tri;
	const std::vector<std::vector<int> >& vtri = __paw.__vtri;

	for(int i = 0; i < __shape.nPoints(); i++)
	{
		x = 0.0;	y = 0.0;
		k = 0;
		//The only problem with this approach is which triangle do we use?
		//In general there will be several triangles that share the i-th vertex.
		for(k = 0; k < vtri[i].size(); k++)// see Figure (11)
		{
			tri_idx = vtri[i][k];
			v1 = tri[tri_idx][0];
			v2 = tri[tri_idx][1];
			v3 = tri[tri_idx][2];

			AAM_PAW::Warp(S0[2*i],S0[2*i+1],
				__sMean[v1].x, __sMean[v1].y,__sMean[v2].x, __sMean[v2].y,__sMean[v3].x, __sMean[v3].y,
					xw, yw,	S[2*v1], S[2*v1+1], S[2*v2], S[2*v2+1], S[2*v3], S[2*v3+1]);
			x += xw;		y += yw;
		}
		// average the result so as to smooth the warp at each vertex
		SEst[2*i] = x/k;		SEst[2*i+1] = y/k;
	}
}


//============================================================================
void AAM_IC::Draw(IplImage* image, const AAM_Shape& Shape, int type)
{
	if(type == 0) AAM_Common::DrawPoints(image, Shape);
	else if(type == 1) AAM_Common::DrawTriangles(image, Shape, __paw.__tri);
	else if(type == 2) 
	{
		cvGEMM(__error_t, __texture.GetBases(), 1, NULL, 1, __lamda, CV_GEMM_B_T);
		__texture.CalcTexture(__lamda, __warp_t);
		AAM_PAW paw;
		double minV, maxV;
		cvMinMaxLoc(__warp_t, &minV, &maxV);
		cvConvertScale(__warp_t, __warp_t, 255/(maxV-minV), -minV*255/(maxV-minV));
		paw.Train(Shape, __Points, __Storage, __paw.GetTri(), false);
		AAM_Common::DrawAppearance(image, Shape, __warp_t, paw, __paw);
	}
	else LOGW("ERROR(%s, %d): Unsupported drawing type\n", __FILE__, __LINE__);
}


//============================================================================
void AAM_IC::Write(std::ofstream& os)
{
	__shape.Write(os);
	__texture.Write(os);
	__paw.Write(os);

	__sMean.Write(os);
	__sStar1.Write(os); __sStar2.Write(os);
	__sStar3.Write(os); __sStar4.Write(os);

	WriteCvMat(os, __G);
}

//============================================================================
void AAM_IC::Read(std::ifstream& is)
{
	__shape.Read(is);
	__texture.Read(is);
	__paw.Read(is);

	int nPoints = __shape.nPoints();
	__sMean.resize(nPoints);
	__sStar1.resize(nPoints); __sStar2.resize(nPoints);
	__sStar3.resize(nPoints); __sStar4.resize(nPoints);
	__sMean.Read(is);
	__sStar1.Read(is); __sStar2.Read(is);
	__sStar3.Read(is); __sStar4.Read(is);

	__G = cvCreateMat(__shape.nModes()+4, __texture.nPixels(), CV_64FC1);
	ReadCvMat(is, __G);

	//alocate memory for on-line fitting stuff
	__Points = cvCreateMat (1, __shape.nPoints(), CV_32FC2);
	__Storage = cvCreateMemStorage(0);

	__update_s0 = cvCreateMat(1, __shape.nPoints()*2, CV_64FC1);
	__inv_pq = cvCreateMat(1, __shape.nModes()+4, CV_64FC1);
	__warp_t = cvCreateMat(1, __texture.nPixels(), CV_64FC1);
	__error_t = cvCreateMat(1, __texture.nPixels(), CV_64FC1);
	__search_pq = cvCreateMat(1, __shape.nModes()+4, CV_64FC1);
	__delta_pq = cvCreateMat(1, __shape.nModes()+4, CV_64FC1);
	__current_s = cvCreateMat(1, __shape.nPoints()*2, CV_64FC1);
	__update_s = cvCreateMat(1, __shape.nPoints()*2, CV_64FC1);
	__lamda  = cvCreateMat(1, __texture.nModes(), CV_64FC1);
}
