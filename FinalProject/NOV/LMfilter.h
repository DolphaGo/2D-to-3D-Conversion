#include "Segmentation.h"
const int sup = 19;
/*
2D Img : src
3D Img : dst
*/
void convF(Mat src, Mat dst, int size) {
	int w = src.cols;
	int h = src.rows;
	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			dst.at<float>(size, y, x) = src.at<float>(y, x);
		}
	}
}
Mat gaussian1d(float sigma, Mat x, int ord) {
	float var = sigma * sigma;
	Mat g1(sup, sup, CV_32F);
	Mat g;
	for (int i = 0; i < sup; i++)
		for (int j = 0; j < sup; j++)
			g1.at<float>(i, j) = (1 / sqrt(2 * CV_PI*var)*exp(-1 * x.at<float>(0, i*sup + j) / (2 * var)));
	if (ord == 0) {
		g = g1;
	}
	else if (ord == 1) {
		for (int i = 0; i < sup; i++)
			for (int j = 0; j < sup; j++)
				g1.at<float>(i, j) = -g1.at<float>(i, j) * ((x.at<float>(0, i*sup + j)) / var);
		g = g1;
	}
	else {
		for (int i = 0; i < sup; i++)
			for (int j = 0; j < sup; j++)
				g1.at<float>(i, j) = -g1.at<float>(i, j) * ((x.at<float>(0, i*sup + j))*(x.at<float>(0, i*sup + j)) - var) / (var*var);
		g = g1;
	}
	return g;
}
Mat gaussian(float scale) {
	Mat dst(sup, sup, CV_32F);
	int hsup = (sup - 1) / 2;

	for (int i = 0; i < sup; i++) {
		for (int j = 0; j < sup; j++) {
			dst.at<float>(i, j) = exp(-((i - hsup)*(i - hsup) + (j - hsup)*(j - hsup)) / (2 * scale*scale)) / (2 * CV_PI*scale*scale);
		}
	}
	return dst;
}
Mat LOG(float scale) {
	int hsup = (sup - 1) / 2;
	Mat dst(sup, sup, CV_32F);
	for (int i = 0; i < sup; i++) {
		for (int j = 0; j < sup; j++) {
			dst.at<float>(i, j) = ((i - hsup)*(i - hsup) + (j - hsup)*(j - hsup) - 2 * scale*scale) / (scale*scale*scale*scale)*exp(-1 * ((i - hsup)*(i - hsup) + (j - hsup)*(j - hsup)) / (2 * scale*scale));
		}
	}
	return dst;
}
Mat makefilter(float scale, float phasex, float phasey, Mat pts) {
	Mat x = pts.row(0);
	Mat y = pts.row(1);
	Mat gx = gaussian1d(3 * scale, x, phasex);
	Mat gy = gaussian1d(scale, y, phasey);
	Mat temp = gx.mul(gy);
	Mat dst(sup, sup, CV_32F);
	for (int y = 0; y < sup; y++) {
		for (int x = 0; x < sup; x++) {
			dst.at<float>(y, x) = temp.at<float>(0, y*sup + x);
		}
	}
	//normalize(dst, dst, -1, 1, NORM_MINMAX);
	return dst;
}

Mat makeLMfilter() {
	float SCALEX = sqrt(2);
	int NORIENT = 6;
	int NBAR = 6;
	int NEDGE = 6;
	int NROTINV = 3;
	int NF = NBAR + NEDGE + NROTINV;
	int size[3] = { NF,sup,sup };
	int hsup = (sup - 1) / 2;
	Mat dst(3, size, CV_32F);
	Mat orgpts(2, sup*sup, CV_32S);
	Mat rotpts(2, sup*sup, CV_32F);

	for (int i = 0; i < sup; i++) {
		for (int j = 0; j < sup; j++)
		{
			orgpts.at<int>(0, i * sup + j) = i - hsup;
			orgpts.at<int>(1, i * sup + j) = hsup - j;
		}
	}
	int count = 0;
	for (int i = 0; i < NORIENT; i++) {
		float angle = CV_PI* i / NORIENT;
		float cosin = cos(angle);
		float s = sin(angle);
		for (int j = 0; j < sup*sup; j++) {
			rotpts.at<float>(0, j) = cosin*orgpts.at<int>(0, j) - s*orgpts.at<int>(1, j);
			rotpts.at<float>(1, j) = s*orgpts.at<int>(0, j) + cosin*orgpts.at<int>(1, j);
		}
		Mat temp;
		temp = makefilter(SCALEX, 0, 1, rotpts);
		convF(temp, dst, count);

		temp = makefilter(SCALEX, 0, 2, rotpts);
		convF(temp, dst, count + NEDGE);

		count++;
	}
	count = NBAR + NEDGE;
	Mat temp;
	temp = gaussian(SCALEX);
	convF(temp, dst, count);
	count++;
	temp = LOG(SCALEX);
	convF(temp, dst, count);
	count++;
	temp = LOG(SCALEX*SCALEX);
	convF(temp, dst, count);
	count++;

	return dst;
}
Mat calc(Mat src, Mat filter) {
	Mat gImg;
	cvtColor(src, gImg, COLOR_BGR2GRAY);
	int w = src.cols;
	int h = src.rows;
	int size[] = { 15,h,w };
	Mat dst(3, size, filter.type());
	int hsup = (sup - 1) / 2;
	for (int count = 0; count < 15; count++) {
		Mat temp(h, w, CV_32F, Scalar(0));
		for (int y = 0; y < h; y++) {
			for (int x = 0; x < w; x++) {
				if (x >= hsup && x < w - hsup && y >= hsup && y < h - hsup) {
					for (int i = 0; i < sup; i++) {
						for (int j = 0; j < sup; j++) {
							temp.at<float>(y, x) += filter.at<float>(count, i, j)*gImg.at<uchar>(y - hsup + i, x - hsup + j);
						}
					}
				}
				dst.at<float>(count, y, x) = temp.at<float>(y, x);
			}
		}
	}
	return dst;
}