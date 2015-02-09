package me.greatyao;

import org.opencv.core.Mat;

public class AAMFit {
	
	static{  
		// Load native library after(!) OpenCV initialization
        System.loadLibrary("aamlibrary");
		}  

	
	public AAMFit(){}
	
	//Step 1: load aamlibrary model
	public static native boolean nativeReadModel(String modelName);

	//Step 2: load viola-jones face detect cascade model
	public static native boolean nativeInitCascadeDetector(String cascadeName);

	//Step 3: detect face
	public static boolean detectOne(Mat imageGray, Mat face){
		return nativeDetectOne(imageGray.getNativeObjAddr(), face.getNativeObjAddr());
	}
	
	//Step 3.x: init face for fitting
	public static void initShape(Mat faces){
		nativeInitShape(faces.getNativeObjAddr());
	}
	
	//Step 4: fitting now
	public static boolean fitting(Mat imageGray, Mat shapes, long n_iteration){
		return nativeFitting(imageGray.getNativeObjAddr(), shapes.getNativeObjAddr(), n_iteration);
	}
	
	//Step 5: draw result
	public static void drawImage(Mat image, Mat shape, long type){
		nativeDrawImage(image.getNativeObjAddr(), shape.getNativeObjAddr(), type);
	}
	
	private static native boolean nativeDetectAll(long inputImage, long faces);
	private static native boolean nativeDetectOne(long inputImage, long face);
	private static native void nativeInitShape(long faces);		
	private static native boolean nativeFitting(long inputImage, long shapes, long n_iteration);
	private static native void nativeDrawImage(long inputImage, long shape, long type);
}
