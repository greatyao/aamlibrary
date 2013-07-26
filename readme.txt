/****************************************************************************
*                                 AAMLibrary                                *
* Copyright (c) 2008 by Yao Wei, all rights reserved.                       *
* Contact:     	njustyw@gmail.com                                           *
*                                                                           *
****************************************************************************/



===================
System Requirements
===================

AAMLibrary is written to be platform independent. It has been tested on Windows XP.
AAMLibrary makes use of the OpenCV library version 1.0 that can be downloaded from:

                http://sourceforge.net/projects/opencv
 
In order to make it work, you must modify a bug in OpenCV and rebuild it:
	-- cvBoundingRect() doesn't work fine. 
	-- opencv/cv/src/cvshapedescr.cpp around line 1161 
	-- replace CV_32SC1 to CV_32SC2 and CV_32FC1 to CV_32FC2.

================
A Quick Tutorial
================

To get familiared quickly with the library, execute the following steps:

1. Train the Cootes's basic active appearance models using 3 parymid levels:
   > build -t 0 -p 3 ../image jpg pts haarcascade_frontalface_alt2.xml model1

2. Train the Matthews and S. Baker's Inverse Compositional models using 2 parymid levels:
   > build -t 1 -p 2 ../image jpg pts haarcascade_frontalface_alt2.xml model2

3. Test fitting/Alignment:
   > fit my.amf haarcascade_frontalface_alt2.xml test.jpg/test.avi

Notice: to make the program run faster, you are suggested to work in the release version.
 
====================
For more details, you can go to the directory "example" to see some demos.
===================


===================
Questions?
===================

For further questions or bug reporting, please email me.

/****************************************************************************
*                 Enjoy this compact library!                               *
****************************************************************************/





