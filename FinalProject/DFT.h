#pragma once

#include "opencv2\opencv.hpp"
using namespace cv;

#define N 128
#define BLK 16

void BWDFT(Mat img, float* result) {
	resize(img, img, Size(N, N));
	int count = 0;
	for (int y = 0; y < N; y += BLK) {
		for (int x = 0; x < N; x += BLK) {
			Mat temp(BLK, BLK, CV_32FC1);
			//Img 조각내기
			for (int yy = y; yy < y + BLK; yy++)
				for (int xx = x; xx < x + BLK; xx++)
					temp.at<float>(yy - y, xx - x) = img.at<uchar>(yy, xx)*cos(CV_PI*(xx - x)*(yy - y) / (BLK*BLK))*cos(CV_PI*(xx - x)*(yy - y) / (BLK*BLK));
			//조각낸 이미지 DFT하기
			Mat planes[] = { Mat_<float>(temp),Mat::zeros(temp.size(),CV_32F) };
			Mat complexI;
			merge(planes, 2, complexI);
			dft(complexI, complexI);
			split(complexI, planes);
			Mat magI;
			magnitude(planes[0], planes[1], magI);
			
			//magI += Scalar::all(1);
			//log(magI, magI);


			//magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));

			//int cx = magI.cols / 2;
			//int cy = magI.rows / 2;

			//Mat q0(magI, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
			//Mat q1(magI, Rect(cx, 0, cx, cy));  // Top-Right
			//Mat q2(magI, Rect(0, cy, cx, cy));  // Bottom-Left
			//Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right

			//Mat tmp;
			//q0.copyTo(tmp);
			//q3.copyTo(q0);
			//tmp.copyTo(q3);
			////q0과 q3의 위치를 바꿈
			//q1.copyTo(tmp);
			//q2.copyTo(q1);
			//tmp.copyTo(q2);
			////q1과 q2의 위치를 바꿈

			//normalize(magI, magI, 0,1, CV_MINMAX);

			// 결과값 저장하기
			for (int yy = y; yy < y + BLK; yy++)
				for (int xx = x; xx < x + BLK; xx++)
					result[yy*N + xx] = magI.at<float>(yy - y, xx - x);
		}
	}
}