#include "resize.h"

#define MAX 1000000
#define scaled_size 128
Point stack[MAX];
int top;
float laplacian[9] = { 0,1,0,1,-4,1,0,1,0 };

void init_stack(void)
{
	top = -1;
}

int stack_is_full(void)
{
	if (top > MAX - 1) return 1;
	else return 0;
}

int stack_is_empty(void)
{
	if (top < 0) return 1;
	else return 0;
}

Point push(Point t)
{
	if (top >= MAX - 1)
	{
		printf("Stack overflow!\n");
		return Point(0, 0);
	}
	stack[++top] = t;
	return t;
}

Point pop()
{
	if (top < 0)
	{
		printf("Stack underflow!!\n");
		return Point(0, 0);
	}
	return stack[top--];
}

Point peek()
{
	if (top < 0)
	{
		printf("Stack underflow!!\n");
		return Point(0, 0);
	}
	return stack[top];
}

int chk(Mat input, int x, int y)
{
	if (input.at<uchar>(y, x) != 0) return 1;
	else return 0;
}

Mat saliency_cut(char* filename)
{
	Mat img;
	Mat saliency_map;
	Mat original = imread(filename, 1);
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

	merge(planes, 2, complexImg); // planes[0]= A+jB, merge�� 1�������� ���ļ� 2������ �����. 
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
			if (intensity > 2 * average)
				objectmap.at<float>(y, x) = 1;
			else
				objectmap.at<float>(y, x) = 0;
		}
	}
	//////////////////////////saliency///////////////////////////


	//Object seed map�� Background seed map�� ����ڴ�.
	Mat L1o(saliency_map.rows, saliency_map.cols, CV_32FC1);
	Mat L1b(saliency_map.rows, saliency_map.cols, CV_32FC1);
	int cx = saliency_map.cols / 2;
	int cy = saliency_map.rows / 2;

	//������ ���� saliency_map�� objectmap�� width�� height�� ���� .

	for (int y = 0; y < objectmap.rows; y++) {
		for (int x = 0; x < objectmap.cols; x++) {
			if (objectmap.at<float>(y, x) == 1 && sqrt((cx - x)*(cx - x) + (cy - y)*(cy - y)) <= 0.4*objectmap.cols)
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
			if (objectmap.at<float>(y, x) == 0 && sqrt((cx - x)*(cx - x) + (cy - y)*(cy - y)) > 0.45*objectmap.cols &&  sqrt((cx - x)*(cx - x) + (cy - y)*(cy - y)) < 0.5*objectmap.cols)
			{
				L1b.at<float>(y, x) = 1;
			}
			else
			{
				L1b.at<float>(y, x) = 0;
			}
		}
	}

	resize(img, img, Size(original.cols, original.rows));
	resize(saliency_map, saliency_map, Size(original.cols, original.rows));
	resize(objectmap, objectmap, Size(original.cols, original.rows));
	resize(L1o, L1o, Size(original.cols, original.rows));
	resize(L1b, L1b, Size(original.cols, original.rows));
	/////////////////////////////////////////SIZE��ȯ///////////////////

	/////////////////////////////graph cut/////////////////////////////

	Mat mask(original.rows, original.cols, CV_8UC1);

	mask.setTo(GC_PR_FGD);
	Rect rect = Rect(cx - (int)0.3*original.cols, cy - (int)0.3*original.rows, (int)0.6*original.cols, (int)0.6*original.rows);

	/*rect.x = max(0, (int)(cx - 0.3*original.cols));
	rect.y = max(0, (int)(cy - 0.3*original.rows));
	rect.width = min(rect.width, (int)0.6*original.cols);
	rect.height = min(rect.height, (int)0.6*original.rows);*/
	(mask(rect)).setTo(Scalar(GC_PR_FGD));

	for (int y = 0; y < objectmap.rows; y++) {
		for (int x = 0; x < objectmap.cols; x++)
		{
			if (L1o.at<float>(y, x) == 1) mask.at<uchar>(y, x) = GC_PR_FGD;
			if (L1b.at<float>(y, x) == 1) mask.at<uchar>(y, x) = GC_BGD;

		}
	}
	for (int y = 0; y < objectmap.rows; y++) {
		for (int x = 0; x < objectmap.cols; x++)
		{
			if (0.05*objectmap.cols > x || 0.95*objectmap.cols < x) mask.at<uchar>(y, x) = GC_BGD;
		}
	}
	Mat bgdModel, fgdModel;
	//cout << mask;
	grabCut(original, mask, rect, bgdModel, fgdModel, 10, GC_INIT_WITH_MASK);
	//// and copy all the foreground-pixels to a temporary image
	Mat1b mask_fgpf = (mask == GC_FGD) | (mask == GC_PR_FGD);

	//compare(mask, GC_PR_FGD, mask, CMP_EQ);
	Mat3b tmp = Mat3b::zeros(objectmap.rows, objectmap.cols);
	original.copyTo(tmp, mask_fgpf);
	Mat tmp2(tmp.rows, tmp.cols, CV_8UC1);
	//PDM �����
	cvtColor(tmp, tmp2, COLOR_BGR2GRAY);
	Mat CCA(tmp.rows, tmp.cols, CV_8UC1, Scalar(0));

	//CCA Recursion

	init_stack(); //���� ����ְ�
	int label_num = 0;
	for (int y = 1; y < tmp.rows - 1; y++)
	{
		for (int x = 1; x < tmp.cols - 1; x++)
		{
			if (tmp2.at<uchar>(y, x) == 0 || CCA.at<uchar>(y, x) != 0) continue; //�̹����� ��ġ�� ���� �ʰų�, ��ŷ�� �� ���¶�� �ѱ���
			label_num++; //1���� �����Ұ��� ��ŷ
			Point t;
			t.x = x; t.y = y; //�̰� ���� ���͸��� ��ģ�ֵ� ��, OV����,
			push(t); //push push 

			while (!stack_is_empty()) //������ �������� ����Ұž�
			{
				int ky = peek().y;//�ϴ� ���� �ҷ���
				int kx = peek().x;//���� �ҷ��԰�
				pop();//�� �����ʹ� ������ 

				CCA.at<uchar>(ky, kx) = label_num; //�׸��� ���� ��ŷ�� �Ұǵ�

				for (int ny = ky - 1; ny <= ky + 1; ny++)
				{
					if (ny < 0 || ny >= tmp.rows) continue;
					for (int nx = kx - 1; nx <= kx + 1; nx++)
					{
						if (nx < 0 || nx >= tmp.cols) continue;

						//�� �� �ڵ尡 �߿��� �û����� �����ִ°�, ���� �ȼ��� �������� 8���� �˻��ϴ°���

						if (tmp2.at<uchar>(ny, nx) == 0 || CCA.at<uchar>(ny, nx) != 0) 
							continue; //�̰͵� ���� �ߴ��� ó��, ���͸��ϴ°��� 
																								 //�̹� ��ŷ�� �Ǿ��ְų� OV�� �ƴϸ� ����
						push(Point(nx, ny));//����� �ɻ縦 ��ġ�� ���Ƴ� �ֵ��� �ֺ��ȼ��̶�� �Ű� �װ͵� push push

						CCA.at<uchar>(ny, nx) = label_num;//���� ������ �󺧸��Ұ���
					}
				}
			}//�̰� ������ �������� �Ұž� �̹��� ������ ���� ��ü�� ov�� 0,0���� �̹��� ������ �ѹ��� ������
			 //������ �� ���, ���� �ٽ� ���� �ȼ��� �Ѿ�� �󺧹�ȣ �ø��� ���ο� ��ŷ�� �����ϴ°���

		}
	}
	Mat OV_PDM(original.rows, original.cols, CV_8UC1, Scalar(0));
	int max; //������ ���� �̹����� �츮�� �˰��ִ� �����̶� �޶� min��� max�� ä������   
	for (int k = 1; k <= label_num; k++)
	{
		max = 0;

		for (int y = 0; y < original.rows; y++)
		{
			for (int x = 0; x < original.cols; x++)
			{
				if (CCA.at<uchar>(y, x) == k)
				{
					if (max < y)
					{
						max = y;
					}
				}
			}
		}
		//�� �󺧵� ��ü�� ���ؼ� ���� ���� ��ȣ�� ã����
		for (int y = 0; y < original.rows; y++)
		{
			for (int x = 0; x < original.cols; x++)
			{
				if (CCA.at<uchar>(y, x) == k) // �� k�� �Ǿ��ִ� ���� ó���ϰ���.
				{
					OV_PDM.at<uchar>(y, x) = 255 * max / original.rows;
				}
			}
		}
	}

	//�� �ܴ� �׶��̼�~
	for (int y = 0; y < original.rows; y++)
	{
		for (int x = 0; x < original.cols; x++)
		{
			if (OV_PDM.at<uchar>(y, x) != 0) continue;

			OV_PDM.at<uchar>(y, x) = 128 * y / original.rows;
		}
	}

	img.release();

	Mat Gimg(original.rows, original.cols, CV_8UC1);
	cvtColor(original, Gimg, COLOR_BGR2GRAY);
	Mat Btemp(original.rows, original.cols, CV_8UC1, Scalar(0));
	//Laplacian(Gimg, Btemp, CV_8UC1,3);

	//   imshow("Btemp", Btemp);
	GaussianBlur(Gimg, Btemp, Size(3, 3), 8, 8);
	Mat B(original.rows, original.cols, CV_8UC1);
	for (int y = 0; y < original.rows; y++)
	{
		for (int x = 0; x < original.cols; x++)
		{
			B.at<uchar>(y, x) = Gimg.at<uchar>(y, x) - Btemp.at<uchar>(y, x);
		}
	}

	Mat OV_FDM(original.rows, original.cols, CV_8UC1, Scalar(0));
	Mat sum_up(original.rows, original.cols, CV_32FC1, Scalar(0));

	//// show it
	//char savename[256];
	//   sprintf(savename, "initial_PDM\\");
	//   strcat(savename, filename);

	imshow("saliency_map", saliency_map);
	//imshow("original", original);
	imshow("object_map", objectmap);
	imshow("L1o ", L1o);
	imshow("L1b", L1b);
	//imshow("tmp", tmp);
	imshow("initial_PDM", OV_PDM);
	waitKey(0);
	//imwrite(savename, OV_PDM);
	OV_PDM.convertTo(OV_PDM, CV_32F);
	/*char namename[256] = "scalesize16seedB32\\";
	strcat(namename, filename);
	imwrite(namename, tmp);*/
	//Output.release();
	return OV_PDM;
}

Mat saliency_cut_mat(Mat original, Mat &objectmap, Mat &saliency_map)
{
	Mat img;
	//Mat saliency_map;
	//Mat original = imread(filename, 1);
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

	merge(planes, 2, complexImg); // planes[0]= A+jB, merge�� 1�������� ���ļ� 2������ �����. 
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

	//Mat objectmap(saliency_map.rows, saliency_map.cols, CV_32F);

	float intensity = 0;
	for (int y = 0; y < objectmap.rows; y++) {
		for (int x = 0; x < objectmap.cols; x++) {
			intensity = saliency_map.at<float>(y, x);
			if (intensity > 16 * average)
				objectmap.at<float>(y, x) = 1;
			else
				objectmap.at<float>(y, x) = 0;
		}
	}
	//////////////////////////saliency///////////////////////////


	//Object seed map�� Background seed map�� ����ڴ�.
	Mat L1o(saliency_map.rows, saliency_map.cols, CV_32FC1);
	Mat L1b(saliency_map.rows, saliency_map.cols, CV_32FC1);
	int cx = saliency_map.cols / 2;
	int cy = saliency_map.rows / 2;

	//������ ���� saliency_map�� objectmap�� width�� height�� ���� .

	for (int y = 0; y < objectmap.rows; y++) {
		for (int x = 0; x < objectmap.cols; x++) {
			//if (objectmap.at<float>(y, x) == 1 && sqrt((cx - x)*(cx - x) + (cy - y)*(cy - y)) <= 0.38*objectmap.cols)
			if(objectmap.at<float>(y,x)==1 && x>(int)(0.1*objectmap.cols)&& x<(int)(0.9*objectmap.cols))
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
			//if (objectmap.at<float>(y, x) == 0 && sqrt((cx - x)*(cx - x) + (cy - y)*(cy - y)) > 0.45*objectmap.cols &&  sqrt((cx - x)*(cx - x) + (cy - y)*(cy - y)) < 0.5*objectmap.cols)
			if (objectmap.at<float>(y, x) == 1 && x>(int)(0.1*objectmap.cols) && x<(int)(0.9*objectmap.cols))
			{
				L1b.at<float>(y, x) = 1;
			}
			else
			{
				L1b.at<float>(y, x) = 0;
			}
		}
	}

	resize(img, img, Size(original.cols, original.rows));
	resize(saliency_map, saliency_map, Size(original.cols, original.rows));
	resize(objectmap, objectmap, Size(original.cols, original.rows));
	resize(L1o, L1o, Size(original.cols, original.rows));
	resize(L1b, L1b, Size(original.cols, original.rows));
	/////////////////////////////////////////SIZE��ȯ///////////////////

	/////////////////////////////graph cut/////////////////////////////

	Mat mask(original.rows, original.cols, CV_8UC1);
	mask.setTo(GC_PR_FGD);
	Rect rect = Rect(0, 0, original.cols, original.rows);

	/*rect.x = max(0, (int)(cx - 0.3*original.cols));
	rect.y = max(0, (int)(cy - 0.3*original.rows));
	rect.width = min(rect.width, (int)0.6*original.cols);
	rect.height = min(rect.height, (int)0.6*original.rows);*/
	(mask(rect)).setTo(Scalar(GC_PR_FGD));


	for (int y = 0; y < objectmap.rows; y++) {
		for (int x = 0; x < objectmap.cols; x++)
		{
			if (L1o.at<float>(y, x) == 1) mask.at<uchar>(y, x) = GC_PR_FGD;
			if (L1b.at<float>(y, x) == 1) mask.at<uchar>(y, x) = GC_BGD;

		}
	}

	for (int y = 0; y < objectmap.rows; y++) {
		for (int x = 0; x < objectmap.cols; x++)
		{
			if (0.1*objectmap.cols > x || 0.9*objectmap.cols < x) mask.at<uchar>(y, x) = GC_BGD;
		}
	}
	Mat bgdModel, fgdModel;
	//cout << mask;
	grabCut(original, mask, rect, bgdModel, fgdModel, 10, GC_INIT_WITH_MASK);
	//// and copy all the foreground-pixels to a temporary image
	Mat1b mask_fgpf = (mask == GC_FGD) | (mask == GC_PR_FGD);

	//compare(mask, GC_PR_FGD, mask, CMP_EQ);
	Mat3b tmp = Mat3b::zeros(objectmap.rows, objectmap.cols);
	original.copyTo(tmp, mask_fgpf);
	Mat tmp2(tmp.rows, tmp.cols, CV_8UC1);

	//PDM �����
	cvtColor(tmp, tmp2, COLOR_BGR2GRAY);
	Mat CCA(tmp.rows, tmp.cols, CV_32S, Scalar(0));

	//CCA Recursion

	vector<Point> vec;

	int label_num = 0;
	for (int y = 1; y < tmp.rows - 1; y++)
	{
		for (int x = 1; x < tmp.cols - 1; x++)
		{
			if (tmp2.at<uchar>(y, x) == 0 || CCA.at<int>(y, x) != 0)
				continue; //�̹����� ��ġ�� ���� �ʰų�, ��ŷ�� �� ���¶�� �ѱ���
			label_num++; //1���� �����Ұ��� ��ŷ
			Point t;
			t.x = x;
			t.y = y; //�̰� ���� ���͸��� ��ģ�ֵ� ��, OV����,
			vec.push_back(t); //push push 

			while (!vec.empty()) //������ �������� ����Ұž�
			{
				int ky = vec.back().y;
				int kx = vec.back().x;
				//cout << "ky:"<< ky <<",  kx:" << kx << "label :" << label_num << endl;
				vec.pop_back();//�� �����ʹ� ������ 

				CCA.at<int>(ky, kx) = label_num; //�׸��� ���� ��ŷ�� �Ұǵ�

				for (int ny = ky - 1; ny <= ky + 1; ny++)
				{
					if (ny < 0 || ny >= tmp.rows) continue;
					for (int nx = kx - 1; nx <= kx + 1; nx++)
					{
						if (nx < 0 || nx >= tmp.cols) continue;

						//�� �� �ڵ尡 �߿��� �û����� �����ִ°�, ���� �ȼ��� �������� 8���� �˻��ϴ°���

						if (tmp2.at<uchar>(ny, nx) == 0 || CCA.at<int>(ny, nx) != 0)
							continue;

						//�̰͵� ���� �ߴ��� ó��, ���͸��ϴ°��� 
						//�̹� ��ŷ�� �Ǿ��ְų� OV�� �ƴϸ� ����

						vec.push_back(Point(nx, ny));//����� �ɻ縦 ��ġ�� ���Ƴ� �ֵ��� �ֺ��ȼ��̶�� �Ű� �װ͵� push push
						CCA.at<int>(ny, nx) = label_num;//���� ������ �󺧸��Ұ���

					}
				}
			}
			//�̰� ������ �������� �Ұž� �̹��� ������ ���� ��ü�� ov�� 0,0���� �̹��� ������ �ѹ��� ������
			//������ �� ���, ���� �ٽ� ���� �ȼ��� �Ѿ�� �󺧹�ȣ �ø��� ���ο� ��ŷ�� �����ϴ°���
		}
	}
	Mat OV_PDM(original.rows, original.cols, CV_8UC1, Scalar(0));
	int max; //������ ���� �̹����� �츮�� �˰��ִ� �����̶� �޶� min��� max�� ä������   
	for (int k = 1; k <= label_num; k++)
	{
		max = 0;

		for (int y = 0; y < original.rows; y++)
		{
			for (int x = 0; x < original.cols; x++)
			{
				if (CCA.at<int>(y, x) == k)
				{
					if (max < y)
					{
						max = y;
					}
				}
			}
		}
		//�� �󺧵� ��ü�� ���ؼ� ���� ���� ��ȣ�� ã����
		for (int y = 0; y < original.rows; y++)
		{
			for (int x = 0; x < original.cols; x++)
			{
				if (CCA.at<int>(y, x) == k) // �� k�� �Ǿ��ִ� ���� ó���ϰ���.
				{
					OV_PDM.at<uchar>(y, x) = 255 * max / original.rows;
				}
			}
		}
	}

	//�� �ܴ� �׶��̼�~
	for (int y = 0; y < original.rows; y++)
	{
		for (int x = 0; x < original.cols; x++)
		{
			if (OV_PDM.at<uchar>(y, x) != 0) continue;

			OV_PDM.at<uchar>(y, x) = 128 * y / original.rows;
		}
	}

	/*imshow("saliency_map", saliency_map);
	imshow("object_map", objectmap);
	imshow("L1o ", L1o);
	imshow("L1b", L1b);
	imshow("tmp", tmp);
	waitKey(0);*/

	OV_PDM.convertTo(OV_PDM, CV_32F);
	normalize(OV_PDM, OV_PDM, 0, 1, CV_MINMAX);

	return OV_PDM;
}