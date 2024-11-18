#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <stdio.h>
#include <cmath>

using namespace cv;
using namespace std;

#define M_PI 3.14159265358979323846

void GaussianFilter(double GFilter[][3]) {
	double sigma = 20.0;
	double denominator = 2.0 * sigma * sigma;

	double sum = 0.0;

	//Gaussian Filter 수식 적용
	for (int x = -1; x <= 1; x++) {
		for (int y = -1; y <= 1; y++) {
			double r_mol = (x * x) + (y * y);
			GFilter[x + 1][y + 1] = (exp(-(r_mol / denominator))) / (M_PI * denominator);
			sum += GFilter[x + 1][y + 1];
		}
	}

	for (int i = 0; i < 3; ++i)
		for (int j = 0; j < 3; ++j)
			GFilter[i][j] /= sum;
}

int main() {
	//원본 이미지
	Mat grayimg = imread("Lenna.png", IMREAD_GRAYSCALE);
	
	// 최종 이미지를 저장한 Mat 변수
	Mat output;

	//Gaussian Filter로 사용할 2차원 배열을 3*3 형태로 선언
	double GFilter[3][3];
	GaussianFilter(GFilter);

	int row = grayimg.rows;
	int col = grayimg.cols;

	//새로운 mat을 row와 col을 2칸씩 늘려서 만들고 0으로 채워준다.
	Mat padding;
	padding.create(row + 2, col + 2, grayimg.type());
	padding.setTo(Scalar::all(0));

	//패딩 부분을 제외한 곳에 원본이미지의 픽셀을 넣는다.
	//0~511까지의 픽셀을 순서대로 넣어준다.
	//이때 패딩이 포함된 Mat은 가로 1~512 세로 1~512까지 밀어 넣어야하므로 i+1, j+1
	for (int i = 0; i <= row - 1; i++) {
		for (int j = 0; j <= col - 1; j++) {
			padding.at<uchar>(i + 1, j + 1) += grayimg.at<uchar>(i, j);
		}
	}

	output.create(grayimg.size(), grayimg.type());
	output.setTo(Scalar::all(0));

	//Gaussian Filter 적용
	for (int i = 0; i <= row - 1; i++) {
		for (int j = 0; j <= col - 1; j++) {
			for (int k = 0; k <= 2; k++) {
				for (int l = 0; l <= 2; l++) {
					output.at<uchar>(i, j) += padding.at<uchar>(i + k, j + l) * GFilter[k][l];
				}
			}
		}
	}

	imshow("origin", grayimg);
	imshow("padding", padding);
	imshow("output", output);

	imwrite("padding.jpg", padding);
	imwrite("output.jpg", output);

	waitKey(0);

	return 0;
}