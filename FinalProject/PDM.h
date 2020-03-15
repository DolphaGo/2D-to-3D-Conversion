#include "OV\OVRendering.h"
#include "NOV\NOVSVM.h"
#include "NOV\FDM.h"
#include "NOV\NOVRendering.h"
#define alpha 0.95


void OVprocess(Mat src,int count) {
	char filename[256];
	char depthmap[256];
	Mat initialPDM(src.rows, src.cols, CV_32F);
	Mat FDM(src.rows, src.cols, CV_32F);
	Mat FinalPDM(src.rows, src.cols, CV_32F);
	Mat objectmap(scaled_size, scaled_size, CV_32F);
	Mat saliencymap(scaled_size, scaled_size, CV_32F);
	cout << "initialPDM" << endl;

	initialPDM = saliency_cut_mat(src,objectmap,saliencymap); //segment the image

	cout << "FDM" << endl;
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
	objectmap.convertTo(objectmap, CV_8U, 255);
	saliencymap.convertTo(saliencymap, CV_8U, 255);
	//sprintf(depthmap, "VideoImage\\OV\\128,128\\o%d.jpg", count);
	//imwrite(depthmap, objectmap);
	sprintf(depthmap, "VideoImage\\OV\\128\\d%d.jpg", count);
	imwrite(depthmap, initialPDM);
	//sprintf(depthmap, "VideoImage\\OV\\128\\s%d.jpg", count);
	//imwrite(depthmap, saliencymap);
	sprintf(filename, "VideoImage\\OV\\128\\r%d.jpg", count);
	imwrite(filename, result);
	
	//printf("OVSVAVE!!\n");
}

void NOVProcess(Mat img, int count)
{
	char filename[256];
	char depthmap[256];
	vector<vector<Pixel>> sp;
	int num = segmentImage(img, sp);
	Mat nov = NOVPredict(img, sp, num);
	Mat fdm = FDM(img, sp);
	Mat depthMap(img.rows, img.cols, CV_32F);
	fusion(nov, fdm, depthMap);
	Mat render = Rendering(img, depthMap, 100, 5);
	depthMap.convertTo(depthMap, CV_8U, 255);

	sprintf(depthmap, "VideoImage\\NOV\\d%d.jpg", count);
	imwrite(depthmap, depthMap);
	sprintf(filename, "VideoImage\\NOV\\r%d.jpg", count);
	imwrite(filename, render);
}

