#include <Windows.h>
#include <opencv2/photo.hpp>
#include <string>
#include <conio.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>
#include <math.h>
#include <time.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"

#define pi 3.141592653589793238462643383279502884197169399375105820974944592307816406286
#define KK 8
#define BLK 128/KK
#define N BLK*KK
#define numOfImg 2000
#define OV 1
#define NOV -1

using namespace cv;
using namespace cv::ml;
using namespace std;
