#pragma once
#include "opencv2\opencv.hpp"
#include "segment-image.h"
#include <vector>

// minimum number of pixels for an individual blob in segmentation
#define MIN_SEGMENTATION_BLOB_SIZE 100  
// sigma smoothing value when running graph image segmentation
#define SIGMA 0.5
// c is the value of threshold
#define c 100
using namespace cv;

image<rgb>* convertMatToNativeImage(Mat input) {
	int w = input.cols;
	int h = input.rows;
	image<rgb> *im = new image<rgb>(w, h);
	for (int i = 0; i<w; i++) {
		for (int j = 0; j<h; j++) {
			rgb curr;
			curr.r = input.at<Vec3b>(j, i)[2];
			curr.g = input.at<Vec3b>(j, i)[1];
			curr.b = input.at<Vec3b>(j, i)[0];
			im->data[i + j*w] = curr;
		}
	}
	return im;
}


Mat convertNativeToMatImage(image<rgb>* input) {
	int w = input->width();
	int h = input->height();
	Mat output(h, w, CV_8UC3);
	for (int i = 0; i<w; i++) {
		for (int j = 0; j<h; j++) {
			rgb curr = input->data[i + j*w];
			output.at<Vec3b>(j, i)[2] = curr.r;
			output.at<Vec3b>(j, i)[1] = curr.g;
			output.at<Vec3b>(j, i)[0] = curr.b;
		}
	}
	return output;
}

int segmentImage(Mat input, vector<vector<Pixel>> &resultBuffer) {
	image<rgb> * converted = convertMatToNativeImage(input);
	int number;
	image<rgb> *rgbImg = segment_image(converted, SIGMA, c, MIN_SEGMENTATION_BLOB_SIZE, &number, resultBuffer);
	//Mat output = convertNativeToMatImage(rgbImg);
	
	//cout << "number of superpixel:" << number << endl;
//	resize(output, output, output.size() / 2);
//	imshow("superpixel", output);
	
	//output.release();
	delete rgbImg;
	delete converted;
	return number;
}
