#pragma once
#include "Segmentation.h"
#include "segment-image.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>

#include "LBP.h"

#define testImg 7
#define dim 76
#define pi 3.141592
#define numOfImg 53
#define threshold -0.8

#define MATH_MIN3(x,y,z)		( (y) <= (z) ? ((x) <= (y) ? (x) : (y)) : ((x) <= (z) ? (x) : (z)) )
#define MATH_MAX3(x,y,z)		( (y) >= (z) ? ((x) >= (y) ? (x) : (y)) : ((x) >= (z) ? (x) : (z)) )


using namespace cv::ml;
struct hsv_color {
	unsigned char h;        // Hue: 0 ~ 255 (red:0, gree: 85, blue: 171)
	unsigned char s;        // Saturation: 0 ~ 255
	unsigned char v;        // Value: 0 ~ 255
};


hsv_color RGB2HSV(unsigned char r, unsigned char g, unsigned char b)
{
	unsigned char rgb_min, rgb_max;

	rgb_min = MATH_MIN3(b, g, r);
	rgb_max = MATH_MAX3(b, g, r);

	hsv_color hsv;
	hsv.v = rgb_max;
	if (hsv.v == 0) {
		hsv.h = hsv.s = 0;
		return hsv;
	}

	hsv.s = 255 * (rgb_max - rgb_min) / hsv.v;
	if (hsv.s == 0) {
		hsv.h = 0;
		return hsv;
	}

	if (rgb_max == r) {
		hsv.h = 0 + 43 * (g - b) / (rgb_max - rgb_min);
	}
	else if (rgb_max == g) {
		hsv.h = 85 + 43 * (b - r) / (rgb_max - rgb_min);
	}
	else /* rgb_max == rgb.b */ {
		hsv.h = 171 + 43 * (r - g) / (rgb_max - rgb_min);
	}

	return hsv;
}

void MultiSVM() {
	char a[256];
	char b[256];
	Mat trainDataMat;
	Mat labels;
	int skynum = 0, verticalnum = 0, groundnum = 0;
	for (int img = 1; img <= numOfImg; img++) {
		sprintf(a, "NOVImage\\scene%d.jpg", img);
		sprintf(b, "NOVImage\\scene%d.bmp", img);
		Mat bmp = imread(b);
		Mat src = imread(a);
		Mat LBP = getLBPMat(src);
		vector<vector<Pixel>> sp;
		int num = segmentImage(src, sp);

		vector<vector<Pixel>>::iterator it;

		//Set labels for SVM
		int i = 0;

		for (it = sp.begin(); it != sp.end(); it++) {
			vector<Pixel>::iterator it2;
			int label;
			float r = 0, g = 0, b = 0, count = 0;
			float rmean = 0, gmean = 0, bmean = 0, xmean = 0, ymean = 0;
			float hmean = 0, smean = 0, vmean = 0;
			float lbp[59] = { 0 };
			float huehistogram[5] = { 0 };
			float saturationhistogram[3] = { 0 };

			for (it2 = sp.at(i).begin(); it2 != sp.at(i).end(); it2++) {
				hsv_color hsv = RGB2HSV(src.at<Vec3b>(it2->y, it2->x)[2], src.at<Vec3b>(it2->y, it2->x)[1], src.at<Vec3b>(it2->y, it2->x)[0]);
				r += bmp.at<Vec3b>(it2->y, it2->x)[2];
				g += bmp.at<Vec3b>(it2->y, it2->x)[1];
				b += bmp.at<Vec3b>(it2->y, it2->x)[0];
				hmean += hsv.h;
				smean += hsv.s;
				vmean += hsv.v;
				int huebin = (int)hsv.h / (255 / 5);
				int saturationbin = (int)hsv.s / (255 / 3);
				huehistogram[huebin] ++;
				saturationhistogram[saturationbin] ++;
				rmean += src.at<Vec3b>(it2->y, it2->x)[2];
				gmean += src.at<Vec3b>(it2->y, it2->x)[1];
				bmean += src.at<Vec3b>(it2->y, it2->x)[0];
				xmean += it2->x;
				ymean += it2->y;
				count++;
				lbp[(int)LBP.at<uchar>(it2->y, it2->x)] += 1;
			}

			hmean /= (count * 255); smean /= (count * 255); vmean /= (count * 255);
			r /= count; g /= count; b /= count;
			rmean /= (count * 255); gmean /= (count * 255); bmean /= (count * 255);
			xmean /= (count*bmp.cols); ymean /= (count*bmp.rows);
			for (int k = 0; k < 5; k++)
				huehistogram[k] /= count;
			for (int k = 0; k < 3; k++)
				saturationhistogram[k] /= count;

			for (int k = 0; k < 59; k++)
				lbp[k] /= count;

			Mat temp(1, dim, CV_32F);

			temp.at<float>(0, 0) = rmean;
			temp.at<float>(0, 1) = gmean;
			temp.at<float>(0, 2) = bmean;
			temp.at<float>(0, 3) = hmean;
			temp.at<float>(0, 4) = smean;
			temp.at<float>(0, 5) = vmean;
			temp.at<float>(0, 6) = 0;// xmean;
			temp.at<float>(0, 7) = ymean;
			temp.at<float>(0, 8) = count;
			for (int k = 0; k < 59; k++)
				temp.at<float>(0, k + 9) = lbp[k];
			for (int k = 0; k < 5; k++)
				temp.at<float>(0, 68 + k) = huehistogram[k];
			for (int k = 0; k < 3; k++)
				temp.at<float>(0, 73 + k) = saturationhistogram[k];
			//Sky
			if (r > 200 && g < 70 && b < 70) {
				label = 1;
				labels.push_back(label);
				trainDataMat.push_back(temp);
				skynum++;
				/*
				Point p;
				for (it2 = src.at(i).begin(); it2 != src.at(i).end(); it2++) {
				p.x = it2->x; p.y = it2->y;
				circle(img, p, 1, Scalar(0, 0, 255));
				}
				*/
			}
			//Vertical
			if (r < 70 && g >200 && b < 70) {
				label = 2;
				labels.push_back(label);
				trainDataMat.push_back(temp);
				verticalnum++;
				/*
				Point p;
				for (it2 = src.at(i).begin(); it2 != src.at(i).end(); it2++) {
				p.x = it2->x; p.y = it2->y;
				circle(img, p, 1, Scalar(0, 255, 0));
				}
				*/
			}
			if (r < 70 && g < 70 && b > 200) {
				label = 3;
				labels.push_back(label);
				trainDataMat.push_back(temp);
				groundnum++;
				/*Point p;
				for (it2 = src.at(i).begin(); it2 != src.at(i).end(); it2++) {
				p.x = it2->x; p.y = it2->y;
				circle(img, p, 1, Scalar(255,0, 0));
				}*/
			}
			i++;
		}
	}
	cout << "SKY:" << skynum << endl << "Vertical" << verticalnum << endl << "Ground" << groundnum << endl;
	Mat1i SkyLabel = (labels != 1) / 255;
	Mat1i VerticalLabel = (labels != 2) / 255;
	Mat1i GroundLabel = (labels != 3) / 255;

	// SKyClass
	Ptr<TrainData> SKY = TrainData::create(trainDataMat, ROW_SAMPLE, SkyLabel);
	Ptr<SVM> SkySvm = SVM::create();
	SkySvm->setType(SVM::C_SVC);
	SkySvm->setKernel(SVM::CHI2);
	SkySvm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, (int)1e3, 1e-6));

	cout << "start SKY SVM train" << endl;
	SkySvm->trainAuto(SKY);
	cout << "complete SKY SVM train" << endl;
	SkySvm->save("SkyClassTrainData.txt");
	cout << "complete SKY SVM save" << endl;

	// Vertical Class
	Ptr<TrainData> Vertical = TrainData::create(trainDataMat, ROW_SAMPLE, VerticalLabel);
	Ptr<SVM> VerticalSvm = SVM::create();
	VerticalSvm->setType(SVM::C_SVC);
	VerticalSvm->setKernel(SVM::CHI2);
	VerticalSvm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, (int)1e3, 1e-6));

	cout << "start Vertical SVM train" << endl;
	VerticalSvm->trainAuto(Vertical);
	cout << "complete Vertical SVM train" << endl;
	VerticalSvm->save("VerticalClassTrainData.txt");
	cout << "complete Vertical SVM save" << endl;

	// Ground Class
	Ptr<TrainData> GROUND = TrainData::create(trainDataMat, ROW_SAMPLE, GroundLabel);
	Ptr<SVM> GroundSvm = SVM::create();
	GroundSvm->setType(SVM::C_SVC);
	GroundSvm->setKernel(SVM::CHI2);
	GroundSvm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, (int)1e3, 1e-6));

	cout << "start GROUND SVM train" << endl;
	GroundSvm->trainAuto(GROUND);
	cout << "complete GROUND SVM train" << endl;
	GroundSvm->save("GroundClassTrainData.txt");
	cout << "complete GROUND SVM save" << endl;

}

void SVMTrain() {
	char a[256];
	char b[256];
	Mat trainDataMat;
	Mat labels;
	int skynum = 0, verticalnum = 0, groundnum = 0;
	for (int img = 1; img <= numOfImg; img++) {
		sprintf(a, "trainset\\NOVImage (%d).jpg", img);
		sprintf(b, "trainset\\NOVImage (%d).bmp", img);
		Mat bmp = imread(b);
		Mat src = imread(a);
		Mat LBP = getLBPMat(src);

		vector<vector<Pixel>> sp;
		int num = segmentImage(src, sp);

		vector<vector<Pixel>>::iterator it;

		//Set labels for SVM
		int i = 0;

		for (it = sp.begin(); it != sp.end(); it++) {
			vector<Pixel>::iterator it2;
			int label;
			float r = 0, g = 0, b = 0, count = 0;
			float rmean = 0, gmean = 0, bmean = 0, xmean = 0, ymean = 0;
			float hmean = 0, smean = 0, vmean = 0;
			float lbp[59] = { 0 };
			float huehistogram[5] = { 0 };
			float saturationhistogram[3] = { 0 };

			for (it2 = sp.at(i).begin(); it2 != sp.at(i).end(); it2++) {
				hsv_color hsv = RGB2HSV(src.at<Vec3b>(it2->y, it2->x)[2], src.at<Vec3b>(it2->y, it2->x)[1], src.at<Vec3b>(it2->y, it2->x)[0]);
				hmean += hsv.h;
				smean += hsv.s;
				vmean += hsv.v;
				int huebin = (int)hsv.h / (255 / 5);
				int saturationbin = (int)hsv.s / (255 / 3);
				huehistogram[huebin] += 1;
				saturationhistogram[saturationbin] += 1;
				r += bmp.at<Vec3b>(it2->y, it2->x)[2];
				g += bmp.at<Vec3b>(it2->y, it2->x)[1];
				b += bmp.at<Vec3b>(it2->y, it2->x)[0];
				rmean += src.at<Vec3b>(it2->y, it2->x)[2];
				gmean += src.at<Vec3b>(it2->y, it2->x)[1];
				bmean += src.at<Vec3b>(it2->y, it2->x)[0];
				xmean += it2->x;
				ymean += it2->y;
				count++;
				lbp[(int)LBP.at<uchar>(it2->y, it2->x)] += 1;
			}
			hmean /= (count * 255); smean /= (count * 255); vmean /= (count * 255);
			rmean /= (count * 255); gmean /= (count * 255); bmean /= (count * 255);
			r /= count; g /= count; b /= count;
			xmean /= (count*src.cols); ymean /= (count*src.rows);
			for (int k = 0; k < 5; k++)
				huehistogram[k] /= count;
			for (int k = 0; k < 3; k++)
				saturationhistogram[k] /= count;

			for (int k = 0; k < 59; k++)
				lbp[k] /= count;

			Mat temp(1, dim, CV_32F);

			temp.at<float>(0, 0) = rmean;
			temp.at<float>(0, 1) = gmean;
			temp.at<float>(0, 2) = bmean;
			temp.at<float>(0, 3) = hmean;
			temp.at<float>(0, 4) = smean;
			temp.at<float>(0, 5) = vmean;
			temp.at<float>(0, 6) = ymean;
			temp.at<float>(0, 7) = ymean;
			temp.at<float>(0, 8) = count / (bmp.cols * bmp.rows);
			for (int k = 0; k < 59; k++)
				temp.at<float>(0, k + 9) = lbp[k];
			for (int k = 0; k < 5; k++)
				temp.at<float>(0, 68 + k) = huehistogram[k];
			for (int k = 0; k < 3; k++)
				temp.at<float>(0, 73 + k) = saturationhistogram[k];

			//Sky
			if (r > 200 && g < 70 && b < 70) {
				label = 1;
				labels.push_back(label);
				trainDataMat.push_back(temp);
				skynum++;
				/*
				Point p;
				for (it2 = src.at(i).begin(); it2 != src.at(i).end(); it2++) {
				p.x = it2->x; p.y = it2->y;
				circle(img, p, 1, Scalar(0, 0, 255));
				}
				*/
			}
			//Vertical
			if (r < 70 && g >200 && b < 70) {
				label = 2;
				labels.push_back(label);
				trainDataMat.push_back(temp);
				verticalnum++;
				/*
				Point p;
				for (it2 = src.at(i).begin(); it2 != src.at(i).end(); it2++) {
				p.x = it2->x; p.y = it2->y;
				circle(img, p, 1, Scalar(0, 255, 0));
				}
				*/
			}
			if (r < 70 && g < 70 && b > 200) {
				label = 3;
				labels.push_back(label);
				trainDataMat.push_back(temp);
				groundnum++;
				/*Point p;
				for (it2 = src.at(i).begin(); it2 != src.at(i).end(); it2++) {
				p.x = it2->x; p.y = it2->y;
				circle(img, p, 1, Scalar(255,0, 0));
				}*/
			}
			i++;
		}
	}

	cout << "SKY:" << skynum << endl << "Vertical" << verticalnum << endl << "Ground" << groundnum << endl;

	cout << trainDataMat.type() << trainDataMat.size() << endl;
	cout << labels.type() << labels.size() << endl;

	Ptr<TrainData> trainData = TrainData::create(trainDataMat, ROW_SAMPLE, labels);
	Ptr<SVM> svm = SVM::create();
	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::CHI2);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, (int)1e4, 1e-6));

	cout << "start train" << endl;
	svm->trainAuto(trainData);
	cout << "complete train" << endl;
	svm->save("NOVPDMTrain.txt");
	cout << "complete save" << endl;
}

/*
dim 0 : Rmean of RGB space in each superpixel;
dim 1 : Gmean of RGB space in each superpixel;
dim 2 : Bmean of RGB space in each superpixel;
dim 3 : Hmean of HSV space in each superpixel;
dim 4 : Smean of HSV space in each superpixel;
dim 5 : Vmean of HSV space in each superpixel;
dim 6 : Xmean in each superpixel;
dim 7 : Ymean in each superpixel;
dim 8 : num of pixel in each superpixel;
dim 9~67 : LBP 59 dim;

*/

void SVMpredict(Mat predictImg) {
	Ptr<SVM> SkySvm = SVM::create();
	SkySvm = SVM::load("SkyClassTrainData.txt");

	Ptr<SVM> VerticalSvm = SVM::create();
	VerticalSvm = SVM::load("VerticalClassTrainData.txt");

	Ptr<SVM> GroundSvm = SVM::create();
	GroundSvm = SVM::load("GroundClassTrainData.txt");

	vector<vector<Pixel>> sp;
	int num = segmentImage(predictImg, sp);

	float *confidence = (float*)calloc(num, sizeof(float));
	Mat Predict(1, dim, CV_32FC1);
	vector<vector<Pixel>>::iterator it;
	Mat result(predictImg.rows, predictImg.cols, CV_8UC3);
	Mat LBP = getLBPMat(predictImg);

	float *predictFeature = (float*)calloc(dim*num, sizeof(float));

	int count = 0;

	for (it = sp.begin(); it != sp.end(); it++) {
		vector<Pixel>::iterator it2;
		for (it2 = sp.at(count).begin(); it2 != sp.at(count).end(); it2++) {
			hsv_color hsv = RGB2HSV(predictImg.at<Vec3b>(it2->y, it2->x)[2], predictImg.at<Vec3b>(it2->y, it2->x)[1], predictImg.at<Vec3b>(it2->y, it2->x)[0]);

			predictFeature[count * dim + 0] += predictImg.at<Vec3b>(it2->y, it2->x)[2];
			predictFeature[count * dim + 1] += predictImg.at<Vec3b>(it2->y, it2->x)[1];
			predictFeature[count * dim + 2] += predictImg.at<Vec3b>(it2->y, it2->x)[0];

			predictFeature[count * dim + 3] += hsv.h;
			predictFeature[count * dim + 4] += hsv.s;
			predictFeature[count * dim + 5] += hsv.v;

			predictFeature[count * dim + 6] += 0; // it2->x;
			predictFeature[count * dim + 7] += it2->y;
			predictFeature[count * dim + 8] ++;
			predictFeature[count*dim + 9 + (int)LBP.at<uchar>(it2->y, it2->x)] += 1;
		}
		predictFeature[count * dim + 0] /= predictFeature[count * dim + 8];
		predictFeature[count * dim + 1] /= predictFeature[count * dim + 8];
		predictFeature[count * dim + 2] /= predictFeature[count * dim + 8];
		predictFeature[count * dim + 3] /= predictFeature[count * dim + 8];
		predictFeature[count * dim + 4] /= predictFeature[count * dim + 8];
		predictFeature[count * dim + 5] /= predictFeature[count * dim + 8];
		predictFeature[count * dim + 6] /= (predictFeature[count * dim + 8] * predictImg.cols);
		predictFeature[count * dim + 7] /= (predictFeature[count * dim + 8] * predictImg.rows);
		for (int k = 0; k < 59; k++)
			predictFeature[count*dim + 9 + k] /= predictFeature[count*dim + 8];
		for (int j = 0; j < dim; j++)
			Predict.at<float>(0, j) = predictFeature[count*dim + j];

		vector<float> response(3);

		response[0] = (SkySvm->predict(Predict, noArray(), StatModel::RAW_OUTPUT));
		response[1] = (VerticalSvm->predict(Predict, noArray(), StatModel::RAW_OUTPUT));
		response[2] = (GroundSvm->predict(Predict, noArray(), StatModel::RAW_OUTPUT));

		int best_class = distance(response.begin(), max_element(response.begin(), response.end()));
		int second_class;

		if (best_class == 0)
			second_class = (response[1] > response[2] ? 1 : 2);
		else if (best_class == 1)
			second_class = (response[0] > response[2] ? 0 : 2);
		else
			second_class = (response[0] > response[1] ? 0 : 1);

		float best_response = MATH_MAX3(response[0], response[1], response[2]);


		//	cout << "bestClass : " << /*best_class <<*/":"<< best_response << endl;
		float temp = fabs(fabs(response[best_class]) - fabs(response[second_class])) / fabs(response[best_class]);
		//cout << "temp : " << temp << endl;

		confidence[count] = temp < 1 ? temp : 1;
		if (confidence[count] <= threshold)
			cout << "confidence" << confidence[count] << endl;

		for (it2 = sp.at(count).begin(); it2 != sp.at(count).end(); it2++) {
			if (best_class == 0) {
				result.at<Vec3b>(it2->y, it2->x)[0] = predictImg.at<Vec3b>(it2->y, it2->x)[0];
				result.at<Vec3b>(it2->y, it2->x)[1] = 0;
				result.at<Vec3b>(it2->y, it2->x)[2] = 0;
				if (confidence[count] <= threshold) {
					result.at<Vec3b>(it2->y, it2->x)[0] = 255;
					result.at<Vec3b>(it2->y, it2->x)[1] = 255;
					result.at<Vec3b>(it2->y, it2->x)[2] = 255;
				}

			}
			if (best_class == 1) {
				result.at<Vec3b>(it2->y, it2->x)[0] = 0;
				result.at<Vec3b>(it2->y, it2->x)[1] = predictImg.at<Vec3b>(it2->y, it2->x)[1];
				result.at<Vec3b>(it2->y, it2->x)[2] = 0;
				if (confidence[count] <= threshold) {
					result.at<Vec3b>(it2->y, it2->x)[0] = 255;
					result.at<Vec3b>(it2->y, it2->x)[1] = 255;
					result.at<Vec3b>(it2->y, it2->x)[2] = 255;
				}
			}
			if (best_class == 2) {
				result.at<Vec3b>(it2->y, it2->x)[0] = 0;
				result.at<Vec3b>(it2->y, it2->x)[1] = 0;
				result.at<Vec3b>(it2->y, it2->x)[2] = predictImg.at<Vec3b>(it2->y, it2->x)[2];
				if (confidence[count] <= threshold) {
					result.at<Vec3b>(it2->y, it2->x)[0] = 255;
					result.at<Vec3b>(it2->y, it2->x)[1] = 255;
					result.at<Vec3b>(it2->y, it2->x)[2] = 255;
				}
			}
		}
		count++;
	}
	imshow("result", result);
	waitKey();
	free(confidence);
}
Mat NOVslic(Mat predictImg) {
	Ptr<SVM> svm = SVM::create();
	int h = predictImg.rows;
	int w = predictImg.cols;
	svm = SVM::load("NOV\\NOVPDMTrain.txt");
	SLIC slic;
	int numlabels;
	int m_spcount = 700;
	double m_compactness = 10;
	unsigned int *img = (unsigned int *)calloc(h*w, sizeof(unsigned int));
	int *labels = new int[w*h];

	// Mat to unsigned int
	for (int y = 0; y < h; y++)
	{
		for (int x = 0; x < w; x++)
		{
			img[y*w + x] = (uint)predictImg.at<Vec3b>(y, x)[0] + ((uint)predictImg.at<Vec3b>(y, x)[1] << 8) + ((uint)predictImg.at<Vec3b>(y, x)[2] << 16);
		}
	}

	// SLIC segmentation
	slic.DoSuperpixelSegmentation_ForGivenNumberOfSuperpixels(img, w, h, labels, numlabels, m_spcount, m_compactness);

	// Draw results
	//slic.DrawContoursAroundSegments(img, labels, w, h, 0);
	vector<vector<Point>> sp;
	vector<vector<Point>>::iterator it;
	for (int i = 0; i < numlabels; i++) {
		vector<Point> v;
		for (int y = 0; y < h; y++) {
			for (int x = 0; x < w; x++) {
				int count = 0;
				if (labels[y*w + x] == i) {
					Point p(x, y);
					v.push_back(p);
					count++;
				}
			}
		}
		sp.push_back(v);
	}
	Mat Predict(1, dim, CV_32FC1);
	Mat labelMat(predictImg.rows, predictImg.cols, CV_8UC3);
	Mat LBP = getLBPMat(predictImg);

	float *predictFeature = (float*)calloc(dim*numlabels, sizeof(float));

	int count = 0;
	vector<Point>::iterator it2;
	for (it = sp.begin(); it != sp.end(); it++) {
		for (it2 = sp.at(count).begin(); it2 != sp.at(count).end(); it2++) {

			hsv_color hsv = RGB2HSV(predictImg.at<Vec3b>(it2->y, it2->x)[2], predictImg.at<Vec3b>(it2->y, it2->x)[1], predictImg.at<Vec3b>(it2->y, it2->x)[0]);

			predictFeature[count * dim + 0] += predictImg.at<Vec3b>(it2->y, it2->x)[2];
			predictFeature[count * dim + 1] += predictImg.at<Vec3b>(it2->y, it2->x)[1];
			predictFeature[count * dim + 2] += predictImg.at<Vec3b>(it2->y, it2->x)[0];

			predictFeature[count * dim + 3] += hsv.h;
			predictFeature[count * dim + 4] += hsv.s;
			predictFeature[count * dim + 5] += hsv.v;

			predictFeature[count * dim + 6] += it2->y; // it2->x;
			predictFeature[count * dim + 7] += it2->y;
			predictFeature[count * dim + 8] +=1;

			predictFeature[count*dim + 9 + (int)LBP.at<uchar>(it2->y, it2->x)] += 1;

			predictFeature[count*dim + (int)hsv.h / (255 / 5)] += 1;
			predictFeature[count*dim + (int)hsv.s / (255 / 3)] += 1;
		}
		predictFeature[count * dim + 0] /= (predictFeature[count * dim + 8] * 255);
		predictFeature[count * dim + 1] /= (predictFeature[count * dim + 8] * 255);
		predictFeature[count * dim + 2] /= (predictFeature[count * dim + 8] * 255);
		predictFeature[count * dim + 3] /= (predictFeature[count * dim + 8] * 255);
		predictFeature[count * dim + 4] /= (predictFeature[count * dim + 8] * 255);
		predictFeature[count * dim + 5] /= (predictFeature[count * dim + 8] * 255);
		predictFeature[count * dim + 6] /= (predictFeature[count * dim + 8] * predictImg.cols);
		predictFeature[count * dim + 7] /= (predictFeature[count * dim + 8] * predictImg.rows);

		for (int k = 0; k < 59; k++)
			predictFeature[count*dim + 9 + k] /= predictFeature[count*dim + 8];
		for (int k = 0; k < 5; k++)
			predictFeature[count*dim + 68 + k] /= predictFeature[count*dim + 8];
		for (int k = 0; k < 3; k++)
			predictFeature[count*dim + 73 + k] /= predictFeature[count*dim + 8];

		predictFeature[count * dim + 8] /= (predictImg.cols *predictImg.rows);

		for (int j = 0; j < dim; j++)
			Predict.at<float>(0, j) = predictFeature[count*dim + j];

		float response = svm->predict(Predict, noArray(), StatModel::RAW_OUTPUT);
		printf("predict\n");

		for (it2 = sp.at(count).begin(); it2 != sp.at(count).end(); it2++) {
			if (response == 1) {
				labelMat.at<Vec3b>(it2->y, it2->x)[0] = predictImg.at<Vec3b>(it2->y, it2->x)[0];
				labelMat.at<Vec3b>(it2->y, it2->x)[1] = 0;
				labelMat.at<Vec3b>(it2->y, it2->x)[2] = 0;
			}

			if (response == 2) {
				labelMat.at<Vec3b>(it2->y, it2->x)[0] = 0;
				labelMat.at<Vec3b>(it2->y, it2->x)[1] = predictImg.at<Vec3b>(it2->y, it2->x)[1];
				labelMat.at<Vec3b>(it2->y, it2->x)[2] = 0;
			}
			if (response == 3) {
				labelMat.at<Vec3b>(it2->y, it2->x)[0] = 0;
				labelMat.at<Vec3b>(it2->y, it2->x)[1] = 0;
				labelMat.at<Vec3b>(it2->y, it2->x)[2] = predictImg.at<Vec3b>(it2->y, it2->x)[2];
			}
		}
		count++;
	}
	Mat firstdepth(labelMat.rows, labelMat.cols, CV_32F);
	int *yver = new int[w];
	for (int x = 0; x < w; x++) {
		int ytemp = 0;
		for (int y = 0; y < h; y++) {
			if (labelMat.at<Vec3b>(y, x)[0] == 0 && labelMat.at<Vec3b>(y, x)[2] == 0) {
				if (ytemp < y)
					ytemp = y;
				// Vertical
				yver[x] = ytemp;
			}
		}
	}

	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			// VERTICAL
			if (labelMat.at<Vec3b>(y, x)[0] == 0 && labelMat.at<Vec3b>(y, x)[2] == 0)
				firstdepth.at<float>(y, x) = 255 * yver[x] / h;
			// SKY
			if (labelMat.at<Vec3b>(y, x)[1] == 0 && labelMat.at<Vec3b>(y, x)[2] == 0)
				firstdepth.at<float>(y, x) = 0;
			// GOURND
			if (labelMat.at<Vec3b>(y, x)[0] == 0 && labelMat.at<Vec3b>(y, x)[1] == 0)
				firstdepth.at<float>(y, x) = 255 * y / h;
		}
	}

	Mat depth(labelMat.rows, labelMat.cols, CV_32F);
	count = 0;
	printf("depth\n");
	for (it = sp.begin(); it != sp.end(); it++) {
		float temp = 0;
		int cnt = 0;
		for (it2 = sp.at(count).begin(); it2 != sp.at(count).end(); it2++) {
			temp += firstdepth.at<float>(it2->y, it2->x);
			cnt++;
		}
		for (it2 = sp.at(count).begin(); it2 != sp.at(count).end(); it2++)
			depth.at<float>(it2->y, it2->x) = temp / cnt;
		count++;
	}

	delete[]yver;
	normalize(depth, depth, 0, 1, CV_MINMAX);

	Predict.release();
	labelMat.release();
	LBP.release();
	firstdepth.release();
	free(predictFeature);
	return depth;
}

Mat NOVPredict(Mat predictImg,vector<vector<Pixel>> &sp, int num) {
	Ptr<SVM> Svm = SVM::create();
	int h = predictImg.rows;
	int w = predictImg.cols;
	Svm = SVM::load("mamamooNOVTrain.txt");
	//printf("loadsvm\n");
	Mat Predict(1, dim, CV_32FC1);
	Mat labelMat(predictImg.rows, predictImg.cols, CV_8UC3);
	Mat LBP = getLBPMat(predictImg);
	Mat firstdepth(h, w, CV_32F);
	Mat depth(labelMat.rows, labelMat.cols, CV_32F);
	//printf("getLBP\n");
	vector<vector<Pixel>>::iterator it;

	float *predictFeature = new float[dim*num];
	//printf("num :%d", num);
	int count = 0;
	vector<Pixel>::iterator it2;
	//printf("start predict\n");
	for (it = sp.begin(); it != sp.end(); it++) {
		for (it2 = sp.at(count).begin(); it2 != sp.at(count).end(); it2++) {
			hsv_color hsv = RGB2HSV(predictImg.at<Vec3b>(it2->y, it2->x)[2], predictImg.at<Vec3b>(it2->y, it2->x)[1], predictImg.at<Vec3b>(it2->y, it2->x)[0]);

			predictFeature[count * dim + 0] += predictImg.at<Vec3b>(it2->y, it2->x)[2];
			predictFeature[count * dim + 1] += predictImg.at<Vec3b>(it2->y, it2->x)[1];
			predictFeature[count * dim + 2] += predictImg.at<Vec3b>(it2->y, it2->x)[0];

			predictFeature[count * dim + 3] += hsv.h;
			predictFeature[count * dim + 4] += hsv.s;
			predictFeature[count * dim + 5] += hsv.v;

			predictFeature[count * dim + 6] += it2->y; 
			predictFeature[count * dim + 7] += it2->y;
			predictFeature[count * dim + 8] +=1;

			predictFeature[count*dim + 9 + (int)LBP.at<uchar>(it2->y, it2->x)] += 1;
			
			predictFeature[count*dim + 68 + (int)((float)hsv.h /256 * 5)] += 1;
			predictFeature[count*dim + 73 + (int)((float)hsv.s /256 * 3)] += 1;
		}
		predictFeature[count * dim + 0] /= (predictFeature[count * dim + 8] * 255);
		predictFeature[count * dim + 1] /= (predictFeature[count * dim + 8] * 255);
		predictFeature[count * dim + 2] /= (predictFeature[count * dim + 8] * 255);
		predictFeature[count * dim + 3] /= (predictFeature[count * dim + 8] * 255);
		predictFeature[count * dim + 4] /= (predictFeature[count * dim + 8] * 255);
		predictFeature[count * dim + 5] /= (predictFeature[count * dim + 8] * 255);
		predictFeature[count * dim + 6] /= (predictFeature[count * dim + 8] * w);
		predictFeature[count * dim + 7] /= (predictFeature[count * dim + 8] * h);

		for (int k = 0; k < 59; k++)
			predictFeature[count*dim + 9 + k] /= predictFeature[count*dim + 8];
		for (int k = 0; k < 5; k++)
			predictFeature[count*dim + 68 + k] /= predictFeature[count*dim + 8];
		for (int k = 0; k < 3; k++)
			predictFeature[count*dim + 73 + k] /= predictFeature[count*dim + 8];

		predictFeature[count * dim + 8] /= (h*w);

		for (int j = 0; j < dim; j++)
			Predict.at<float>(0, j) = predictFeature[count*dim + j];

		float response = Svm->predict(Predict, noArray());
	
		for (it2 = sp.at(count).begin(); it2 != sp.at(count).end(); it2++) {
			if (response == 1) {
				labelMat.at<Vec3b>(it2->y, it2->x)[0] = predictImg.at<Vec3b>(it2->y, it2->x)[0];
				labelMat.at<Vec3b>(it2->y, it2->x)[1] = 0;
				labelMat.at<Vec3b>(it2->y, it2->x)[2] = 0;
			}

			if (response == 2) {
				labelMat.at<Vec3b>(it2->y, it2->x)[0] = 0;
				labelMat.at<Vec3b>(it2->y, it2->x)[1] = predictImg.at<Vec3b>(it2->y, it2->x)[1];
				labelMat.at<Vec3b>(it2->y, it2->x)[2] = 0;
			}
			if (response == 3) {
				labelMat.at<Vec3b>(it2->y, it2->x)[0] = 0;
				labelMat.at<Vec3b>(it2->y, it2->x)[1] = 0;
				labelMat.at<Vec3b>(it2->y, it2->x)[2] = predictImg.at<Vec3b>(it2->y, it2->x)[2];
			}
		}
		count++;
	}	
	//printf("markingLabel\n");
	int *yver = new int[w];
	for (int x = 0; x < w; x++) {
		int ytemp = 0;
		for (int y = 0; y < h; y++) {
			if (labelMat.at<Vec3b>(y, x)[0] == 0 && labelMat.at<Vec3b>(y, x)[2] == 0) {
				if (ytemp < y)
					ytemp = y;
				// Vertical
				yver[x] = ytemp;
			}
		}
	}
//	printf("firstdepthmap\n");
	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			// VERTICAL
			if (labelMat.at<Vec3b>(y, x)[0] == 0 && labelMat.at<Vec3b>(y, x)[2] == 0)
				firstdepth.at<float>(y, x) = 255*yver[x] / h;
			// SKY
			if (labelMat.at<Vec3b>(y, x)[1] == 0 && labelMat.at<Vec3b>(y, x)[2] == 0)
				firstdepth.at<float>(y, x) = 0;
			// GOURND
			if (labelMat.at<Vec3b>(y, x)[0] == 0 && labelMat.at<Vec3b>(y, x)[1] == 0)
				firstdepth.at<float>(y, x) = 255* y / h;
		}
	}

	count = 0;
	for (it = sp.begin(); it != sp.end(); it++) {
		float temp = 0;
		int cnt = 0;
		for (it2 = sp.at(count).begin(); it2 != sp.at(count).end(); it2++) {
			temp += firstdepth.at<float>(it2->y, it2->x);
			cnt++;
		}
		for (it2 = sp.at(count).begin(); it2 != sp.at(count).end(); it2++)
			depth.at<float>(it2->y, it2->x) = temp / cnt;
		count++;
	}
	//printf("depth\n");
	normalize(depth, depth, 0, 1, CV_MINMAX);

	delete[]yver;
	delete[]predictFeature;

	return depth;
}
