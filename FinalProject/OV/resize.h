#include "OVsvm.h"

#define imgsize 128
void resize_img(char *inputname,char *outputname) {
	Mat input = imread(inputname, 1);
	if (input.empty())
		return;
	imshow("input",input);
	waitKey(1);
	Mat output(imgsize, imgsize, CV_8UC3);
	resize(input, output, Size(imgsize, imgsize));
	imwrite(outputname, output);
	imshow("output", output);
	waitKey(1);
	
}