#include "OV\OVRendering.h"
#include "NOV\NOVSVM.h"
#include "NOV\FDM.h"
#include "NOV\NOVRendering.h"
#define alpha 0.9


void OVprocess(Mat src,int count) {
	char filename[256];
	char depthmap[256];
	Mat initialPDM(src.rows, src.cols, CV_32F);
	Mat FDM(src.rows, src.cols, CV_32F);
	Mat FinalPDM(src.rows, src.cols, CV_32F);

	initialPDM = saliency_cut_mat(src); //segment the image
	FDM = makeFDM_mat(src); // focus depth map of the image
	 //normalize(initialPDM, initialPDM, 0, 1, CV_MINMAX);
	 //normalize(FDM, FDM, 0, 1, CV_MINMAX);

	fusion(initialPDM, FDM, FinalPDM);
	//cout << "fusion" << endl;
	Mat result = Rendering(src, FinalPDM, 100, 5);
//	cout << "Rendering" << endl;
	normalize(initialPDM, initialPDM, 0, 255, CV_MINMAX);
	initialPDM.convertTo(initialPDM, CV_8UC1);
	//video.write(result);
	//sprintf(filename, "momo\\%d.bmp", count);
	sprintf(depthmap, "mist\\OV\\depth%d.jpg", count);
	imwrite(depthmap, initialPDM);
	sprintf(filename, "mist\\OV\\img%d.jpg", count);
	imwrite(filename, result);
	//printf("OVSVAVE!!\n");
}

void NOVProcess(Mat img,int count)
{
	char filename[256];
	char depthmap[256];
	Mat nov =NOVPredict(img);
	
	Mat fdm = FDM(img);
	Mat depthMap = alpha*nov + (1 - alpha)*fdm;
	normalize(depthMap, depthMap, 0, 255, CV_MINMAX);
	depthMap.convertTo(depthMap, CV_8U);
	Mat render = NOVRendering(img, depthMap, 100, 10);
	
	sprintf(depthmap, "lala\\%d.jpg", count);
	imwrite(depthmap, depthMap);
	sprintf(filename, "lala\\%d.bmp", count);
	imwrite(filename, render);
//	printf("NOVSVAVE!!\n");
}
