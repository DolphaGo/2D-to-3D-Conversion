#include "FFT.h"

float trainingData[N*N][numOfImg] = { 0 };

void getlabel() {
	for (int i = 0; i < numOfImg; i++) {
		if (i < numOfImg / 2)
			labels[i] = OV;
		else
			labels[i] = NOV;
	}
}

void DataMapping(float* &data, int cnt)
{
	for (int y = 0; y < N; y++)
	{
		for (int x = 0; x < N; x++)
		{
			trainingData[y*N + x][cnt] = data[y*N + x];
		}
		//printf("\n");
	}
}

Ptr<SVM> training_SVM()
{
	// Set up training data
	Mat trainingDataMat(N*N, numOfImg, CV_32FC1, trainingData);
	Mat labelsMat(1, numOfImg, CV_32SC1, labels);

	//cout << "이건 1번\n" << trainingDataMat << endl; //지금 이게 쓰레기 값이 들어가고 있는것 같지?
	//cout << "이건 2번\n" << labelsMat << endl; //이건 아주 잘 나옴

	//Ptr<TrainData> traindata = TrainData::create(trainingDataMat, COL_SAMPLE, labelsMat);
	Ptr<SVM> svm = SVM::create();

	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::LINEAR);

	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 300, 1e-6));
	//svm->trainAuto(traindata);
	svm->train(trainingDataMat, COL_SAMPLE, labelsMat); //이 부분에서 data가 다 사라짐;

	svm->save("traindata.txt");
	return svm;
}

int Predict(Ptr<SVM> svm, Mat src) //predict만 실행하는 함수
{
	float *data = FFT(src);
	Mat sampleMat(1, N*N, CV_32F);

	for (int y = 0; y < N; y++) {
		for (int x = 0; x < N; x++) {
			sampleMat.at<float>(0, y*N + x) = data[y*N + x];
		}
	}              
	
	// Show the decision regions given by the SVM
	float response = svm->predict(sampleMat, noArray(), 0);
	
	if (response == 1) //OV 
	{
		count_ov++;
	}
	else if (response == -1)//NOV 
	{
		count_nov++;
	}

	sampleMat.release();
	return (int)response;

}