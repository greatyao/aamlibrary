CPPFLAGS = `pkg-config --cflags opencv` -I./ -fPIC -O2 -Wall -g -MD
PROGRAMS = libaamlibrary.so fit build 
LIBS = `pkg-config --libs opencv`

OBJS = AAM_Basic.o  AAM_IC.o AAM_PAW.o  AAM_Shape.o  AAM_Util.o AAM_CAM.o AAM_MovieAVI.o  AAM_PDM.o  AAM_TDM.o  VJfacedetect.o
 
all: $(PROGRAMS)

AAM_Basic.o: AAM_Basic.cpp
	g++  $(CPPFLAGS)  -c -o AAM_Basic.o AAM_Basic.cpp
AAM_IC.o: AAM_IC.cpp
	g++  $(CPPFLAGS)  -c -o AAM_IC.o AAM_IC.cpp
AAM_PAW.o: AAM_PAW.cpp
	g++  $(CPPFLAGS)  -c -o AAM_PAW.o AAM_PAW.cpp
AAM_Shape.o: AAM_Shape.cpp
	g++  $(CPPFLAGS)  -c -o AAM_Shape.o AAM_Shape.cpp
AAM_Util.o: AAM_Util.cpp
	g++  $(CPPFLAGS)  -c -o AAM_Util.o AAM_Util.cpp
AAM_CAM.o: AAM_CAM.cpp
	g++  $(CPPFLAGS)  -c -o AAM_CAM.o AAM_CAM.cpp
AAM_MovieAVI.o: AAM_MovieAVI.cpp
	g++  $(CPPFLAGS)  -c -o AAM_MovieAVI.o AAM_MovieAVI.cpp
AAM_PDM.o: AAM_PDM.cpp
	g++  $(CPPFLAGS)  -c -o AAM_PDM.o AAM_PDM.cpp
AAM_TDM.o: AAM_TDM.cpp
	g++  $(CPPFLAGS)  -c -o AAM_TDM.o AAM_TDM.cpp
VJfacedetect.o: VJfacedetect.cpp
	g++  $(CPPFLAGS)  -c -o VJfacedetect.o VJfacedetect.cpp	
fit.o: ../exe/fit.cpp
	g++  $(CPPFLAGS)  -c -o fit.o ../exe/fit.cpp 
train.o: ../exe/train.cpp
	g++  $(CPPFLAGS)  -c -o train.o ../exe/train.cpp 
libaamlibrary.so: $(OBJS)
	g++ -shared -o libaamlibrary.so $(OBJS)	$(LIBS) -lpthread -lrt

fit: fit.o libaamlibrary.so
	g++ -o fit fit.o $(LIBS) -Wl,-rpath=./ -L./ -laamlibrary
build: train.o libaamlibrary.so
	g++ -o build train.o $(LIBS) -Wl,-rpath=./ -L./ -laamlibrary

clean:
	rm -f *.o $(PROGRAMS)
