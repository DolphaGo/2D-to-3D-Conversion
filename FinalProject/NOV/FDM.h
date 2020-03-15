#include "Segmentation.h"
//NOV
Mat FDM(Mat src, vector<vector<Pixel>> sp ) {
	Mat gImg;
	cvtColor(src, gImg, COLOR_BGR2GRAY);
	gImg.convertTo(gImg, CV_32F);
	Mat B;

	GaussianBlur(gImg, B, Size(3, 3), 0.5);
	
	B = gImg - B;
	Mat C;
	normalize(B, C, 0, 1, NORM_MINMAX);
	Mat Df(B.rows,B.cols,CV_32F);

	vector<vector<Pixel>>::iterator it;
	int i = 0;
	for (it = sp.begin(); it != sp.end(); it++) {
		vector<Pixel>::iterator it2;
		float temp = 0;
		int count = 0;
		for (it2 = sp.at(i).begin(); it2 != sp.at(i).end(); it2++) {
			temp += log(fabs(B.at<float>(it2->y, it2->x)) + 1);
			count++;
		}
		temp /= count;

		for (it2 = sp.at(i).begin(); it2 != sp.at(i).end(); it2++)
			Df.at <float > (it2->y, it2->x) = temp;
		i++;
	}

	normalize(Df, Df, 0, 1, NORM_MINMAX);
	return Df;
}