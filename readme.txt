/****************************************************************************
*                                 AAMLibrary                                *
* Copyright (c) 2008-2015 by Yao Wei, all rights reserved.                       *
* Contact:     	njustyw@gmail.com                                           *
*                                                                           *
****************************************************************************/

===================
System Requirements
===================
* opencv 1.0 or later
* cmake 2.6 or later
	
===================
How to build your program
===================

```
$ mkdir build 
$ cd build
$ cmake ..
$ make
```

================
A Quick Tutorial
================

To get familiared quickly with the library, execute the following steps:

1. Train the Cootes's basic active appearance models using 3 parymid levels:
   $ build -t 0 -p 3 ../image jpg pts haarcascade_frontalface_alt2.xml model1

2. Train the Matthews and S. Baker's Inverse Compositional models using 2 parymid levels:
   $ build -t 1 -p 2 ../image jpg pts haarcascade_frontalface_alt2.xml model2

3. Test fitting/Alignment:
   $ fit my.amf haarcascade_frontalface_alt2.xml test.jpg/test.avi
 
For more details, you can go to the directory "example" to see some demos.

For further questions or bug reporting, please email me.





