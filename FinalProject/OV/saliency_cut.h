#include "resize.h"

void saliency_cut_erosion(char* filename)
{
	Mat img;
	Mat saliency_map;
	Mat original = imread(filename, 1);
	const int scaled_size = 64;
	resize(original, img, Size(scaled_size, scaled_size));
	cvtColor(img, img, COLOR_BGR2GRAY);

	Mat planes[] = { Mat_<float>(img), Mat::zeros(img.size(), CV_32F) };

	Mat complexImg;
	merge(planes, 2, complexImg); // Add to the expanded another plane with zeros
	dft(complexImg, complexImg);  // this way the result may fit in the source matrix
	split(complexImg, planes); // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))

	Mat mag, logmag, smooth, spectralResidual;
	magnitude(planes[0], planes[1], mag);
	// compute the magnitude and switch to logarithmic scale
	// => log(sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
	log(mag, logmag);
	boxFilter(logmag, smooth, -1, Size(3, 3));
	subtract(logmag, smooth, spectralResidual);
	exp(spectralResidual, spectralResidual);

	// real part 
	planes[0] = planes[0].mul(spectralResidual) / mag; //mul=multiply, 
													   // imaginary part 
	planes[1] = planes[1].mul(spectralResidual) / mag;

	merge(planes, 2, complexImg); // planes[0]= A+jB, merge는 1차원들을 합쳐서 2차원을 만든다. 
	dft(complexImg, complexImg, DFT_INVERSE | DFT_SCALE);
	split(complexImg, planes);
	// get magnitude
	magnitude(planes[0], planes[1], mag);
	// get square of magnitude
	multiply(mag, mag, mag);
	// Gaussian kernel size. ksize.width and ksize.height can differ but they both must be positive and odd

	GaussianBlur(mag, mag, Size(3, 3), 8, 8);
	normalize(mag, saliency_map, 0, 1, CV_MINMAX);

	int height = saliency_map.rows;
	int width = saliency_map.cols;
	float sum = 0;
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			sum += saliency_map.at<float>(y, x);
		}
	}
	float average = sum / (height*width);
	Mat objectmap(saliency_map.rows, saliency_map.cols, CV_32F);

	float intensity = 0;
	for (int y = 0; y < objectmap.rows; y++) {
		for (int x = 0; x < objectmap.cols; x++) {
			intensity = saliency_map.at<float>(y, x);
			if (intensity > 1 * average)
				objectmap.at<float>(y, x) = 1;
			else
				objectmap.at<float>(y, x) = 0;
		}
	}
	//////////////////////////saliency///////////////////////////


	//Object seed map과 Background seed map을 만들겠다.
	Mat L1o(saliency_map.rows, saliency_map.cols, CV_32FC1);
	Mat L1b(saliency_map.rows, saliency_map.cols, CV_32FC1);
	int cx = saliency_map.cols / 2;
	int cy = saliency_map.rows / 2;

	//어차피 현재 saliency_map과 objectmap의 width와 height는 같음 .

	for (int y = 0; y < objectmap.rows; y++) {
		for (int x = 0; x < objectmap.cols; x++) {
			if (objectmap.at<float>(y, x) == 1 && sqrt((cx - x)*(cx - x) + (cy - y)*(cy - y)) < 0.3*objectmap.cols)
			{
				L1o.at<float>(y, x) = 1;
			}
			else
			{
				L1o.at<float>(y, x) = 0;
			}
		}
	}
	int erosion_size = 0;
	Mat element = getStructuringElement(MORPH_ELLIPSE, Size(2 * erosion_size + 1, 2 * erosion_size + 1), Point(erosion_size, erosion_size));
	erode(L1o, L1o, element);

	for (int y = 0; y < objectmap.rows; y++) {
		for (int x = 0; x < objectmap.cols; x++) {
			if (objectmap.at<float>(y, x) == 0 && sqrt((cx - x)*(cx - x) + (cy - y)*(cy - y)) > 0.38*objectmap.cols &&  sqrt((cx - x)*(cx - x) + (cy - y)*(cy - y)) < objectmap.cols / 2)
			{
				L1b.at<float>(y, x) = 1;
			}
			else
			{
				L1b.at<float>(y, x) = 0;
			}
		}
	}
	int dilation_size = 0;
	Mat element2 = getStructuringElement(MORPH_ELLIPSE, Size(2 * dilation_size + 1, 2 * dilation_size + 1), Point(dilation_size, dilation_size));
	dilate(L1b, L1b, element2);


	resize(saliency_map, saliency_map, Size(original.cols, original.rows));
	resize(objectmap, objectmap, Size(original.cols, original.rows));
	resize(L1o, L1o, Size(original.cols, original.rows));
	resize(L1b, L1b, Size(original.cols, original.rows));
	/////////////////////////////////////////SIZE변환///////////////////

	/////////////////////////////graph cut/////////////////////////////

	Mat mask(original.rows, original.cols, CV_8UC1);
	if (!mask.empty())
		mask.setTo(Scalar::all(GC_BGD));
	mask.setTo(GC_PR_BGD);
	Rect rect = Rect(cx - (int)0.3*original.cols, cy - (int)0.3*original.rows, (int)0.6*original.cols, (int)0.6*original.rows);

	/*rect.x = max(0, (int)(cx - 0.3*original.cols));
	rect.y = max(0, (int)(cy - 0.3*original.rows));
	rect.width = min(rect.width, (int)0.6*original.cols);
	rect.height = min(rect.height, (int)0.6*original.rows);*/
	(mask(rect)).setTo(Scalar(GC_PR_FGD));

	for (int y = 0; y < objectmap.rows; y++) {
		for (int x = 0; x < objectmap.cols; x++)
		{
			if (L1b.at<float>(y, x) == 1) mask.at<uchar>(y, x) = GC_PR_BGD;
			if (L1o.at<float>(y, x) == 1) mask.at<uchar>(y, x) = GC_PR_FGD;
		}
	}
	
	Mat bgdModel, fgdModel;
	//cout << mask;
	grabCut(original, mask, rect, bgdModel, fgdModel, 5, GC_INIT_WITH_MASK);
	//// and copy all the foreground-pixels to a temporary image
	Mat1b mask_fgpf = (mask == GC_FGD) | (mask == GC_PR_FGD);

	//compare(mask, GC_PR_FGD, mask, CMP_EQ);
	Mat3b tmp = Mat3b::zeros(objectmap.rows, objectmap.cols);
	original.copyTo(tmp, mask_fgpf);
	//// show it

	imshow("saliency_map__", saliency_map);
	//imshow("original", original);
	imshow("object_map__", objectmap);
	imshow("L1o_ero ", L1o);
	imshow("L1b_dil", L1b);
	imshow("tmp_", tmp);
	waitKey(50000);
	/*char namename[256] = "scalesize16seedB32\\";
	strcat(namename, filename);
	imwrite(namename, tmp);*/
	//Output.release();
}

void saliency_cut(char* filename)
{
	Mat img;
	Mat saliency_map;
	Mat original = imread(filename, 1);
	const int scaled_size = 64;
	resize(original, img, Size(scaled_size, scaled_size));
	cvtColor(img, img, COLOR_BGR2GRAY);

	Mat planes[] = { Mat_<float>(img), Mat::zeros(img.size(), CV_32F) };

	Mat complexImg;
	merge(planes, 2, complexImg); // Add to the expanded another plane with zeros
	dft(complexImg, complexImg);  // this way the result may fit in the source matrix
	split(complexImg, planes); // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))

	Mat mag, logmag, smooth, spectralResidual;
	magnitude(planes[0], planes[1], mag);
	// compute the magnitude and switch to logarithmic scale
	// => log(sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
	log(mag, logmag);
	boxFilter(logmag, smooth, -1, Size(3, 3));
	subtract(logmag, smooth, spectralResidual);
	exp(spectralResidual, spectralResidual);

	// real part 
	planes[0] = planes[0].mul(spectralResidual) / mag; //mul=multiply, 
													   // imaginary part 
	planes[1] = planes[1].mul(spectralResidual) / mag;

	merge(planes, 2, complexImg); // planes[0]= A+jB, merge는 1차원들을 합쳐서 2차원을 만든다. 
	dft(complexImg, complexImg, DFT_INVERSE | DFT_SCALE);
	split(complexImg, planes);
	// get magnitude
	magnitude(planes[0], planes[1], mag);
	// get square of magnitude
	multiply(mag, mag, mag);
	// Gaussian kernel size. ksize.width and ksize.height can differ but they both must be positive and odd

	GaussianBlur(mag, mag, Size(3, 3), 8, 8);
	normalize(mag, saliency_map, 0, 1, CV_MINMAX);

	int height = saliency_map.rows;
	int width = saliency_map.cols;
	float sum = 0;
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			sum += saliency_map.at<float>(y, x);
		}
	}
	float average = sum / (height*width);
	Mat objectmap(saliency_map.rows, saliency_map.cols, CV_32F);

	float intensity = 0;
	for (int y = 0; y < objectmap.rows; y++) {
		for (int x = 0; x < objectmap.cols; x++) {
			intensity = saliency_map.at<float>(y, x);
			if (intensity > 1 * average)
				objectmap.at<float>(y, x) = 1;
			else
				objectmap.at<float>(y, x) = 0;
		}
	}
	//////////////////////////saliency///////////////////////////


	//Object seed map과 Background seed map을 만들겠다.
	Mat L1o(saliency_map.rows, saliency_map.cols, CV_32FC1);
	Mat L1b(saliency_map.rows, saliency_map.cols, CV_32FC1);
	int cx = saliency_map.cols / 2;
	int cy = saliency_map.rows / 2;

	//어차피 현재 saliency_map과 objectmap의 width와 height는 같음 .

	for (int y = 0; y < objectmap.rows; y++) {
		for (int x = 0; x < objectmap.cols; x++) {
			if (objectmap.at<float>(y, x) == 1 && sqrt((cx - x)*(cx - x) + (cy - y)*(cy - y)) < 0.3*objectmap.cols)
			{
				L1o.at<float>(y, x) = 1;
			}
			else
			{
				L1o.at<float>(y, x) = 0;
			}
		}
	}
	
	for (int y = 0; y < objectmap.rows; y++) {
		for (int x = 0; x < objectmap.cols; x++) {
			if (objectmap.at<float>(y, x) == 0 && sqrt((cx - x)*(cx - x) + (cy - y)*(cy - y)) > 0.38*objectmap.cols &&  sqrt((cx - x)*(cx - x) + (cy - y)*(cy - y)) < objectmap.cols / 2)
			{
				L1b.at<float>(y, x) = 1;
			}
			else
			{
				L1b.at<float>(y, x) = 0;
			}
		}
	}
	
	resize(saliency_map, saliency_map, Size(original.cols, original.rows));
	resize(objectmap, objectmap, Size(original.cols, original.rows));
	resize(L1o, L1o, Size(original.cols, original.rows));
	resize(L1b, L1b, Size(original.cols, original.rows));
	/////////////////////////////////////////SIZE변환///////////////////

	/////////////////////////////graph cut/////////////////////////////

	Mat mask(original.rows, original.cols, CV_8UC1);
	if (!mask.empty())
		mask.setTo(Scalar::all(GC_BGD));
	mask.setTo(GC_PR_BGD);
	Rect rect = Rect(cx - (int)0.3*original.cols, cy - (int)0.3*original.rows, (int)0.6*original.cols, (int)0.6*original.rows);

	/*rect.x = max(0, (int)(cx - 0.3*original.cols));
	rect.y = max(0, (int)(cy - 0.3*original.rows));
	rect.width = min(rect.width, (int)0.6*original.cols);
	rect.height = min(rect.height, (int)0.6*original.rows);*/
	(mask(rect)).setTo(Scalar(GC_PR_FGD));

	for (int y = 0; y < objectmap.rows; y++) {
		for (int x = 0; x < objectmap.cols; x++)
		{
			if (L1b.at<float>(y, x) == 1) mask.at<uchar>(y, x) = GC_PR_BGD;
			if (L1o.at<float>(y, x) == 1) mask.at<uchar>(y, x) = GC_PR_FGD;
		}
	}

	Mat bgdModel, fgdModel;
	//cout << mask;
	grabCut(original, mask, rect, bgdModel, fgdModel, 5, GC_INIT_WITH_MASK);
	//// and copy all the foreground-pixels to a temporary image
	Mat1b mask_fgpf = (mask == GC_FGD) | (mask == GC_PR_FGD);

	//compare(mask, GC_PR_FGD, mask, CMP_EQ);
	Mat3b tmp = Mat3b::zeros(objectmap.rows, objectmap.cols);
	original.copyTo(tmp, mask_fgpf);
		
	//// show it
	imshow("saliency_map", saliency_map);
	imshow("original", original);
	imshow("object_map", objectmap);
	imshow("L1o ", L1o);
	imshow("L1b", L1b);
	imshow("tmp", tmp);
	waitKey(50000);
	/*char namename[256] = "scalesize16seedB32\\";
	strcat(namename, filename);
	imwrite(namename, tmp);*/
	//Output.release();
}