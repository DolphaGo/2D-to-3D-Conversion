#pragma once
#ifndef FEATURE
#define FEATURE

#include "Segmentation.h"
#include "segment-image.h"
#include "pnmfile.h"
#define width 128
#define height 128
#define dim 6


/*
	* feature dimension
	* 0~2 rgb mean
	* 3~4 x,y mean
	* 5 pixel number of superpixels
*/

Mat getFeatureVec(Mat src) {
	Mat img;
	resize(src, img, Size(width, height));
	Mat rgbmean(height, width, CV_8UC3);
	
	vector<vector<Pixel>> result;
	int num = segmentImage(img, result);
	Mat dst(dim, num, CV_32F);

	vector<vector<Pixel>>::iterator it;
	float *feature = new float[num*dim];
	int i = 0;
	for (it = result.begin(); it != result.end(); it++) {
		vector<Pixel>::iterator it2;
		for (it2 = result.at(i).begin(); it2 != result.at(i).end(); it2++) {
			feature[i * dim + 0] += img.at<Vec3b>(it2->y, it2->x)[0];
			feature[i * dim + 1] += img.at<Vec3b>(it2->y, it2->x)[1];
			feature[i * dim + 2] += img.at<Vec3b>(it2->y, it2->x)[2];
			feature[i * dim + 3] += it2->x;
			feature[i * dim + 4] += it2->y;
			feature[i * dim + 5] ++;
		}

		feature[i * dim + 0] /= feature[i * dim + 5];
		feature[i * dim + 1] /= feature[i * dim + 5];
		feature[i * dim + 2] /= feature[i * dim + 5];
		feature[i * dim + 3] /= feature[i * dim + 5];
		feature[i * dim + 4] /= feature[i * dim + 5];
		
		//rgb 평균 출력
		/*
		for (it2 = result.at(i).begin(); it2 != result.at(i).end(); it2++) {
			rgbmean.at<Vec3b>(it2->y, it2->x)[0] = feature[i * dim];
			rgbmean.at<Vec3b>(it2->y, it2->x)[1] = feature[i * dim];
			rgbmean.at<Vec3b>(it2->y, it2->x)[2] = feature[i * dim];
		}
		*/
		
		dst.at<float>(0, i) = feature[i*dim + 0];
		dst.at<float>(1, i) = feature[i*dim + 1];
		dst.at<float>(2, i) = feature[i*dim + 2];
		dst.at<float>(3, i) = feature[i*dim + 3];
		dst.at<float>(4, i) = feature[i*dim + 4];
		dst.at<float>(5, i) = feature[i*dim + 5];
		
		cout << i << "번째 superpixel mean of Y:" << feature[i * dim + 4] << endl;
		cout << i << "번째 superpixel pixel number of superpixels :" << feature[i * dim + 5] << endl;
		
		i++;
	}

	delete[] feature;
	return dst;
}
#endif