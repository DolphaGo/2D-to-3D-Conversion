#include "DFT.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>

using namespace cv::ml;
using namespace std;

#define pi 3.141592

#define numOfImg 2000
#define OV 1
#define NOV -1
#include "DFT.h"

Ptr<SVM> SVMtraining() {
	float **trainingData = (float**)calloc(N*N, sizeof(float*));
	for (int i = 0; i < N*N; i++) {
		trainingData[i] = (float*)calloc(numOfImg, sizeof(float));
	}
	float *data = (float*)calloc(N*N, sizeof(float));
	int labels[numOfImg];
	for (int i = 0; i < numOfImg; i++) {
		if (i < numOfImg / 2)
			labels[i] = OV;
		else
			labels[i] = NOV;
	}

	char filename[256];
	int j = 0;
	for (int i = 0; i < numOfImg; i++)
	{
		if (i < numOfImg / 2) {
			sprintf(filename, "dataset\\newNOV_%d.bmp", i + 1);
			Mat img = imread(filename,0);
			BWDFT(img, data);
			for (int y = 0; y<N; y++)
				for (int x = 0; x<N; x++){
					trainingData[y*N + x][i] = data[y*N + x];
			}
		}
		else {
			j++;
			sprintf(filename, "dataset\\newOV_%d.bmp", j);
			Mat img = imread(filename, 0);
			BWDFT(img, data); 
			for (int y = 0; y<N; y++)
				for (int x = 0; x<N; x++)
					trainingData[y*N + x][i] = data[y*N + x];
		}
	}
	printf("start trainSVM\n\n");
	Mat trainingDataMat(numOfImg, N*N, CV_32FC1);
	for (int i = 0; i < numOfImg; i++) {
		for (int j = 0; j < N*N; j++) {
			trainingDataMat.at<float>(i, j) = trainingData[j][i];
		}
	}

	Mat labelsMat(numOfImg, 1, CV_32SC1, labels);
	//cout << "trainingDataMat" << trainingDataMat << endl;
	//	cout << "label" << labelsMat << endl;
	Ptr<TrainData> traindata = TrainData::create(trainingDataMat, ROW_SAMPLE, labelsMat);
	
	Ptr<SVM> svm = SVM::create();
	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::LINEAR);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 1000, 1e-6));

	//	svm->train(trainingDataMat, COL_SAMPLE, labelsMat);
	svm->trainAuto(traindata);
	svm->save("traindata.txt");
	free(data);
	for (int i = 0; i < N*N; i++)
		free(trainingData[i]);
	free(trainingData);
	return svm;
}

int SVMpredicting(Ptr<SVM> svm, Mat img) {
	float* sample_data = (float*)calloc(N*N, sizeof(float));
	BWDFT(img, sample_data);
	Mat SampleMat(1, N*N, CV_32FC1);
	for (int y = 0; y < N; y++)
		for (int x = 0; x < N; x++)
			SampleMat.at<float>(0, y*N + x) = sample_data[y*N + x];
	int response = svm->predict(SampleMat);
	return response;
}
