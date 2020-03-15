#include "PDM.h"

//void main()
//{
//	//getlabel();//앞의 500개는 OV , 뒤에 500개는 NOV
//	//char samplename[256];
//	//float* data;
//	//Mat output(numOfImg, N*N, CV_32FC1);
//	char filename[256];
//	char outputname[256];
//	char outputname2[256];
//	char outputname3[256];
//	//int j = 0;
//
//	////Step1: map을 만드는 구간
//	//for (int i = 0; i < numOfImg; i++)
//	//{
//	//	if (i < numOfImg / 2)
//	//	{
//	//		sprintf(filename, "dataset\\newOV_%d.bmp", i + 1);
//	//		FFT(filename, data);
//	//		DataMapping(data, i);
//	//	}
//	//	else
//	//	{
//	//		j++;
//	//		sprintf(filename, "dataset\\NOV_%d.bmp", j);
//	//		FFT(filename, data);
//	//		DataMapping(data, i);
//	//	}
//	//}
//	//Step2: svm을 함
//	//Ptr<SVM> svm = training_SVM();
//
//	// load the saved data
//	Ptr<SVM> svm = SVM::create();
//	svm = SVM::load("traindata.txt");
//
//	float* sample_data;
//
//	////OV Test
//	//for (int i = 250; i < 500; i++) { //500장의 샘플
//	//	sprintf(filename, "dataset\\testOV_%d.bmp", i + 1);
//	//	FFT(filename, sample_data);//sample_data는 값을 매우 잘받고있음.
//	//	Predict(svm, sample_data); // predict만 실행하는 것임.
//	//							   //printf("predicting testOV_%d.bmp... OV: %d\tNOV:%d\n", i + 1, count_ov, count_nov);
//	//}
//	//float p_ov = (float)count_ov / (count_ov + count_nov) * 100;
//	//printf("OV 제대로 나온 확률 : %3.2f%%\n", p_ov);
//
//	////NOV Test
//	//count_ov = 0; count_nov = 0;
//	//for (int i = 250; i < 500; i++) { //500장의 샘플
//	//	sprintf(filename, "dataset\\testNOV_%d.bmp", i + 1);
//	//	FFT(filename, sample_data);//sample_data는 값을 매우 잘받고있음.
//	//	Predict(svm, sample_data); //여기서 svm은 train할 대상이 되는 data가 되는 것임.
//	//							   //printf("predicting testNOV_%d.bmp... OV: %d\tNOV:%d\n", i + 1, count_ov, count_nov);
//	//}
//	//float p_nov = (float)count_nov / (count_ov + count_nov) * 100;
//	//printf("NOV 제대로 나온 확률 : %3.2f%%\n", p_nov);
//	
//	int count = 0;
//	float alpha = 0.67;
//	for (int i = 250; i < 300; i++) {
//		sprintf(filename, "dataset\\testOV_%d.bmp", i + 1);
//		//sprintf(filename, "dataset\\testOV.jpg");
//		
//		Mat img = imread(filename, 1);
//		Mat initialPDM(img.rows, img.cols, CV_32F);
//		Mat FDM(img.rows, img.cols, CV_32F);
//		Mat FinalPDM(img.rows, img.cols, CV_32F);
//
//		initialPDM=saliency_cut(filename); //segment the image
//		FDM=makeFDM(filename); // focus depth map of the image
//
//		normalize(initialPDM, initialPDM, 0, 1, CV_MINMAX);
//		normalize(FDM, FDM, 0, 1, CV_MINMAX);
//
//		fusion(initialPDM, FDM, FinalPDM);
//
//		Mat result=Rendering(img, FinalPDM, 100,5);
//		sprintf(filename, "3dOV_%d.bmp", i + 1);
//		imwrite(filename, result);
//
//		//imshow("FinalPDM", PDM);		
//		/*imshow("original", img);
//		imshow("initialPDM", initialPDM);
//		imshow("FDM", FDM);
//		imshow("FinalPDM", FinalPDM);
//		imshow("3D", result);
//		waitKey(2000);*/						
//	}		
//	return;
//}

void main() {
	char filename[256];
	char depthmap[256];
	int count = 0;
	VideoCapture cap("OV\\test3D.mp4");
	int frame_width = cap.get(CV_CAP_PROP_FRAME_WIDTH) / 2; // 960
	int frame_height = cap.get(CV_CAP_PROP_FRAME_HEIGHT) / 2; // 540
	Ptr<SVM>ObjectClassifier = SVM::load("OV\\traindata.txt");
	while (1) {
		Mat frame;
		cap >> frame;
		if (frame.empty()) break;
		count++;
		//if (i < 380) continue;
		printf("Saving %dth frame ...\n", count);
		resize(frame, frame, Size(frame_width, frame_height));
		int response = Predict(ObjectClassifier, frame);
		if (response == OV) {
			Mat initialPDM(frame.rows, frame.cols, CV_32F);
			Mat FDM(frame.rows, frame.cols, CV_32F);
			Mat FinalPDM(frame.rows, frame.cols, CV_32F);

			initialPDM = saliency_cut_mat(frame); //segment the image
			FDM = makeFDM_mat(frame); // focus depth map of the image

			//normalize(initialPDM, initialPDM, 0, 1, CV_MINMAX);
			//normalize(FDM, FDM, 0, 1, CV_MINMAX);

			fusion(initialPDM, FDM, FinalPDM);
			Mat result = Rendering(frame, FinalPDM, 100, 5);

			normalize(initialPDM, initialPDM, 0, 255, CV_MINMAX);
			initialPDM.convertTo(initialPDM, CV_8UC1);
			//video.write(result);
			//sprintf(filename, "momo\\%d.bmp", count);
			sprintf(depthmap, "boiler\\%d.jpg", count);
			imwrite(depthmap, initialPDM);
			sprintf(filename, "boiler\\%d.bmp", count);
			imwrite(filename, result);
			printf("OVSVAVE!!\n");
		}
		else {
		}
		//imshow("FinalPDM", FinalPDM);
		//waitKey(1000);
		/*imshow("cap", result);
		char c = (char)waitKey(10);
		if (c == 27)
			break;*/
	}
	cap.release();
	//video.release();
}

