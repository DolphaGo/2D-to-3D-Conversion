#include "Segmentation.h"

const int lookup[256] = {
	0, 1, 2, 3, 4, 58, 5, 6, 7,58,58,58, 8,58, 9,10,
	11,58,58,58,58,58,58,58,12,58,58,58,13,58,14,15,
	16,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,
	17,58,58,58,58,58,58,58,18,58,58,58,19,58,20,21,
	22,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,
	58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,
	23,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,
	24,58,58,58,58,58,58,58,25,58,58,58,26,58,27,28,
	29,30,58,31,58,58,58,32,58,58,58,58,58,58,58,33,
	58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,34,
	58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,
	58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,35,
	36,37,58,38,58,58,58,39,58,58,58,58,58,58,58,40,
	58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,41,
	42,43,58,44,58,58,58,45,58,58,58,58,58,58,58,46,
	47,48,58,49,58,58,58,50,51,52,58,53,54,55,56,57,
};

Mat getLBPMat(Mat srcImage) {
	int height = srcImage.rows;
	int width = srcImage.cols;
	Mat gray;
	Mat LBP(srcImage.rows, srcImage.cols, CV_32S);
	cvtColor(srcImage, gray, COLOR_BGR2GRAY);
	int w = gray.cols;
	int h = gray.rows;
	int pVal, gVal;
	int Bpattern[8], Binary[8];
	for (int by = 1; by < height - 1; by++) {
		for (int bx = 1; bx < width - 1; bx++) {
			pVal = gray.at<uchar>(by, bx);
			Bpattern[0] = gray.at<uchar>(by - 1, bx);
			Bpattern[1] = gray.at<uchar>(by - 1, bx + 1);
			Bpattern[2] = gray.at<uchar>(by, bx + 1);
			Bpattern[3] = gray.at<uchar>(by + 1, bx + 1);
			Bpattern[4] = gray.at<uchar>(by + 1, bx);
			Bpattern[5] = gray.at<uchar>(by + 1, bx - 1);
			Bpattern[6] = gray.at<uchar>(by, bx - 1);
			Bpattern[7] = gray.at<uchar>(by - 1, bx - 1);
			gVal = 0;
			for (int i = 0; i < 8; i++) {
				if (pVal > Bpattern[i])
					Binary[i] = 1;
				else
					Binary[i] = 0;
			}
			for(int i = 0; i < 8; i++)
				gVal += Binary[i] * pow(2, 7 - i);
			LBP.at<int>(by, bx) = lookup[gVal];
		}
	}
	return LBP;
}