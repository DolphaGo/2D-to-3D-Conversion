#include "PDM.h"
#include "SVM.h"

//#define MAIN
#define OVNOVclassify
#define MAIN
//#define RENDERexperiment
#ifndef OVNOVclassify

void main() {
	int count = 0;
	Ptr<SVM>ObjectClassifier = SVM::load("traindata.txt");
	char ImgName[255];
	for(int i=1;i<3125;i++){
		sprintf(ImgName, "mamamoo\\%d.jpg", i);
		Mat frame = imread(ImgName);

		if (frame.empty())
			continue;
		count++;
		int response = SVMpredicting(ObjectClassifier, frame);
		if (response == OV) {
			OVprocess(frame, count);
			printf("OV %d save\n", count);
		}
		else {
			NOVProcess(frame, count);
			printf("NOV %d save\n", count);
		}
		frame.release();
	}
}
#endif // !OVNOVclassify
#ifndef MAIN
void main()
{
	int count = 1;
	//VideoCapture cap("mamamoo.mp4");
	char ImgName[255];
	Mat frame;
	for(int i=1 ; i<=4838;i+=30){
		sprintf(ImgName, "mamamoo\\OV\\OV (%d).jpg",i);
		frame = imread(ImgName);
		if (frame.empty())continue;
		//printf("%d img\n", i);
		OVprocess(frame, count);
		count++;
	}
}
#endif
#ifndef RENDERexperiment
void main() {
	Mat frame;
	frame = imread("mamamoo\\OV\\OV (1285).jpg");
	Mat initialPDM(frame.rows, frame.cols, CV_32F);
	Mat FDM(frame.rows, frame.cols, CV_32F);
	Mat FinalPDM(frame.rows, frame.cols, CV_32F);
	Mat objectmap(scaled_size, scaled_size, CV_32F);
	Mat saliencymap(scaled_size, scaled_size, CV_32F);
	cout << "initialPDM" << endl;

	initialPDM = saliency_cut_mat(frame, objectmap, saliencymap); //segment the image

	cout << "FDM" << endl;
	FDM = makeFDM_mat(frame); // focus depth map of the image
							//normalize(initialPDM, initialPDM, 0, 1, CV_MINMAX);
							//normalize(FDM, FDM, 0, 1, CV_MINMAX);

	fusion(initialPDM, FDM, FinalPDM);
	
	Mat result = Rendering(frame, initialPDM, 100, 5);
	initialPDM.convertTo(initialPDM, CV_8U, 255);
	imwrite("depth.jpg", initialPDM);
	imwrite("Render.jpg", result);
}
#endif