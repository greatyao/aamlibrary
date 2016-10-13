
# Active Appearance Model C++ Library (AAMLibrary)


## Dependencies
- opencv 1.0 or later
- cmake 2.6 or later
 
## How to build your program

  > mkdir build 

  > cd build

  > cmake ..

  > make

## Quick Tutorial

### Prepare: 
- For model training, you should have several pairs of images and annotations. AAMLibrary supports pts and asf format.
- Download the imm dataset from AAM-API's homepage [link: IMM Dataset](http://www2.imm.dtu.dk/pubdb/views/publication_details.php?id=922)
- Download FRANCK dataset annotated by Cootes [link: Cootes's Dataset](http://personalpages.manchester.ac.uk/staff/timothy.f.cootes/tfc_software.html)
- Download helen dataset from this [link: Helen Dataset](http://ibug.doc.ic.ac.uk/resources/facial-point-annotations/)
- You can generate your pts file by using Cootes's am_tools and asf file by using Stagmman's aam-api.
 

### Training 

- Train the Cootes's basic active appearance models using 3 parymid levels
   > ./build -t 0 -p 3 ../helen jpg pts haarcascade_frontalface_alt2.xml basic.amf

- Train the Matthews and S. Baker's Inverse Compositional models using 2 parymid levels
   > ./build -t 1 -p 2 ../helen jpg pts haarcascade_frontalface_alt2.xml ic.amf

 
### Fitting

- Image alignment on an image 
   > ./fit my.amf haarcascade_frontalface_alt2.xml test.jpg
   

- Face tracking on a video file
   > ./fit my.amf haarcascade_frontalface_alt2.xml test.avi

## Question
if you have any question, contact me at njustyw@gmail.com, THANKS.
