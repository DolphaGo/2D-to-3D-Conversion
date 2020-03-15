#include "doha_saliency_cut.h"
#include "SLIC.h"

Mat makeFDM(char *filename) {
	int x, y;
	int height, width;

	Mat input = imread(filename, 1);
	SLIC slic;
	
	height = input.rows;
	width = input.cols;
	Mat inputG;
	cvtColor(input, inputG, CV_RGB2GRAY);
	Mat inputFloat;
	inputG.convertTo(inputFloat, CV_32F);
	Mat B(height, width, CV_32F);
	Mat Gaussian(height, width, CV_32F);

	GaussianBlur(inputFloat, Gaussian, Size(3, 3), 8);
	B = inputFloat - Gaussian;

	int numlabels;
	int m_spcount = 70;
	double m_compactness = 10;
	unsigned int *img = (unsigned int *)calloc(height*width, sizeof(unsigned int));
	int *labels = new int[width*height];

	// Mat to unsigned int
	for (y = 0; y < height; y++)
	{
		for (x = 0; x < width; x++)
		{
			img[y*width + x] = (uint)input.at<Vec3b>(y, x)[0] + ((uint)input.at<Vec3b>(y, x)[1] << 8) + ((uint)input.at<Vec3b>(y, x)[2] << 16);
		}
	}

	// SLIC segmentation
	slic.DoSuperpixelSegmentation_ForGivenNumberOfSuperpixels(img, width, height, labels, numlabels, m_spcount, m_compactness);

	// Draw results
	slic.DrawContoursAroundSegments(img, labels, width, height, 0);

	Mat result(height, width, CV_8UC3);
	for (y = 0; y < height; y++)
	{
		for (x = 0; x < width; x++)
		{
			result.at<Vec3b>(y, x)[0] = img[y * width + x] & 0xff;
			result.at<Vec3b>(y, x)[1] = img[y * width + x] >> 8 & 0xff;
			result.at<Vec3b>(y, x)[2] = img[y * width + x] >> 16 & 0xff;
		}
	}
	/*for (int y = 0; y < height; y++) {
	for (int x = 0; x < width; x++) {
	cout << labels[y*width + x]<<",";
	}
	cout<<endl;
	}*/
	Mat DF = Mat::zeros(Size(width, height), CV_32FC1);
	float temp;
	int count = 0;
	for (int num = 0; num < numlabels; num++) {
		count = 0;
		temp = 0;
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				if (labels[y*width + x] == num) {
					temp += log(fabs(B.at<float>(y, x)) + 1);
					count++;
				}
			}
		}
		temp /= count;
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				if (labels[y*width + x] == num)
					DF.at<float>(y, x) = temp;
			}
		}
	}
	normalize(DF, DF, 0, 1, CV_MINMAX);
	normalize(B, B, 0, 1, CV_MINMAX);
	//cout << numlabels << endl;
	//slic.SaveSuperpixelLabels(labels, width, height,"superpixelLabels","");
	
	//imwrite("DF_float.bmp", DF);
	//imshow("original", input);
	//imshow("B", B);
	//imshow("SLIC", result);
	//imshow("FDM", DF);
	waitKey(0);
	free(img);
	free(labels);
	return DF;
}

Mat makeFDM_mat(Mat input) {
	int x, y;
	int height, width;

	//Mat input = imread(filename, 1);
	SLIC slic;

	height = input.rows;
	width = input.cols;
	Mat inputG;
	cvtColor(input, inputG, CV_RGB2GRAY);
	Mat inputFloat;
	inputG.convertTo(inputFloat, CV_32F);
	Mat B(height, width, CV_32F);
	Mat Gaussian(height, width, CV_32F);

	GaussianBlur(inputFloat, Gaussian, Size(3, 3), 8);
	B = inputFloat - Gaussian;

	int numlabels;
	int m_spcount = 70;
	double m_compactness = 10;
	unsigned int *img = (unsigned int *)calloc(height*width, sizeof(unsigned int));
	int *labels = new int[width*height];

	// Mat to unsigned int
	for (y = 0; y < height; y++)
	{
		for (x = 0; x < width; x++)
		{
			img[y*width + x] = (uint)input.at<Vec3b>(y, x)[0] + ((uint)input.at<Vec3b>(y, x)[1] << 8) + ((uint)input.at<Vec3b>(y, x)[2] << 16);
		}
	}

	// SLIC segmentation
	slic.DoSuperpixelSegmentation_ForGivenNumberOfSuperpixels(img, width, height, labels, numlabels, m_spcount, m_compactness);

	// Draw results
	slic.DrawContoursAroundSegments(img, labels, width, height, 0);

	Mat result(height, width, CV_8UC3);
	for (y = 0; y < height; y++)
	{
		for (x = 0; x < width; x++)
		{
			result.at<Vec3b>(y, x)[0] = img[y * width + x] & 0xff;
			result.at<Vec3b>(y, x)[1] = img[y * width + x] >> 8 & 0xff;
			result.at<Vec3b>(y, x)[2] = img[y * width + x] >> 16 & 0xff;
		}
	}
	/*for (int y = 0; y < height; y++) {
	for (int x = 0; x < width; x++) {
	cout << labels[y*width + x]<<",";
	}
	cout<<endl;
	}*/
	Mat DF = Mat::zeros(Size(width, height), CV_32FC1);
	float temp;
	int count = 0;
	for (int num = 0; num < numlabels; num++) {
		count = 0;
		temp = 0;
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				if (labels[y*width + x] == num) {
					temp += log(fabs(B.at<float>(y, x)) + 1);
					count++;
				}
			}
		}
		temp /= count;
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				if (labels[y*width + x] == num)
					DF.at<float>(y, x) = temp;
			}
		}
	}
	normalize(DF, DF, 0, 1, CV_MINMAX);
	normalize(B, B, 0, 1, CV_MINMAX);
	waitKey(0);
	free(img);
	free(labels);
	return DF;
}