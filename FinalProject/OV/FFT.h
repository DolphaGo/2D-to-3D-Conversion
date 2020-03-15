#include "settings.h"

int labels[numOfImg] = { 0 };
int count_ov = 0;
int count_nov = 0;

float* FFT(Mat img)
{	
	float* data2svm = (float*)calloc(N*N, sizeof(float));

	resize(img, img, Size(N, N));//interpolation이 자동으로 적용


	Mat output(N, N, CV_32FC1); //mag이미지 확인용
	Mat block(BLK, BLK, CV_32FC1); // hanning window

	for (int y = 0; y <= N - BLK; y += BLK) {
		for (int x = 0; x <= N - BLK; x += BLK) {
			for (int yy = y; yy < y + BLK; yy++)
				for (int xx = x; xx < x + BLK; xx++)
					block.at<float>(yy - y, xx - x) = img.at<uchar>(yy, xx)*cos((xx - x)* (yy - y)*pi / (BLK*BLK))*cos((xx - x)* (yy - y)*pi / (BLK*BLK)); //BWDFT

			Mat planes[] = { Mat_<float>(block), Mat::zeros(block.size(), CV_32F) };
			Mat complexI;
			merge(planes, 2, complexI);
			dft(complexI, complexI);
			split(complexI, planes);
			magnitude(planes[0], planes[1], planes[0]);
			// planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
			Mat magI = planes[0];
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
			//q1과 q2의 위치를 바꿈
			//normalize(magI, magI, -pi, pi, NORM_MINMAX);
			for (int yy = y; yy < y + BLK; yy++)
				for (int xx = x; xx < x + BLK; xx++) {
					data2svm[yy*N + xx] = output.at<float>(yy, xx) = magI.at<float>(yy - y, xx - x);
				}
		}
	}
	return data2svm;
}
