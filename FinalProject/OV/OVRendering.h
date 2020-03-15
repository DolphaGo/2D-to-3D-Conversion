#include "FDM.h"

void fusion(Mat& initialPDM, Mat& FDM, Mat& FinalPDM) {
	float alpha = 0.67;
	for (int y = 0; y < initialPDM.rows; y++) {
		for (int x = 0; x < initialPDM.cols; x++) {
			FinalPDM.at<float>(y, x) = alpha * initialPDM.at<float>(y, x) + (1 - alpha)*FDM.at<float>(y, x);
		}
	}
	/*imshow("initial?", initialPDM);
	imshow("FDM?", FDM);
	imshow("FinalPDM", FinalPDM);
	waitKey();*/
}

Mat Rendering(Mat src, Mat DepthMapImg, float V, float D) {
	const float E = 6.35;
	int w = src.cols;
	int h = src.rows;
	
	Mat leftImg(h, w, CV_8UC3);
	Mat rightImg(h, w, CV_8UC3);
	Mat result(h, w, CV_8UC3);

	float w3 = D * E / (D + V);
	float *parallax = (float*)calloc(h*w, sizeof(float));
	float max = 0;
	int pixel = 500;
	float *val = (float*)calloc(h*w, sizeof(float));
	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			val[y*w + x] = DepthMapImg.at<float>(y, x);
			if (max < val[y*w + x])
				max = val[y*w + x];
		}
	}
	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			parallax[y*w + x] = w3 * (val[y*w + x]/(max -2))*pixel;
		}
	}
	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			for (int k = 0; k < 3; k++) {
				if (x - parallax[y*w + x] / 2 >= 0&&x + parallax[y*w + x] / 2 >= 0 && x - parallax[y*w + x] / 2 < w&&x + parallax[y*w + x] / 2 < w){
					rightImg.at<Vec3b>(y, x )[k] = src.at<Vec3b>(y, x - parallax[y*w + x] / 2)[k];
					leftImg.at<Vec3b>(y, x )[k] = src.at<Vec3b>(y, x + parallax[y*w + x] / 2)[k];
				}
				else
				{
					rightImg.at<Vec3b>(y, x)[k] = src.at<Vec3b>(y, x)[k];
					leftImg.at<Vec3b>(y, x)[k] = src.at<Vec3b>(y, x)[k];
				}
			}
		}
	}
	//left :Red right : Blue
	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			result.at<Vec3b>(y, x)[0] = rightImg.at<Vec3b>(y, x)[0];
			result.at<Vec3b>(y, x)[1] = rightImg.at<Vec3b>(y, x)[1];
			result.at<Vec3b>(y, x)[2] = leftImg.at<Vec3b>(y, x)[2];
		}
	}
//	printf("5\n");
	free(parallax);
	free(val);
	return result;
}