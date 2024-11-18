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

	//Gaussian Filter ���� ����
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
	//���� �̹���
	Mat grayimg = imread("Lenna.png", IMREAD_GRAYSCALE);
	
	// ���� �̹����� ������ Mat ����
	Mat output;

	//Gaussian Filter�� ����� 2���� �迭�� 3*3 ���·� ����
	double GFilter[3][3];
	GaussianFilter(GFilter);

	int row = grayimg.rows;
	int col = grayimg.cols;

	//���ο� mat�� row�� col�� 2ĭ�� �÷��� ����� 0���� ä���ش�.
	Mat padding;
	padding.create(row + 2, col + 2, grayimg.type());
	padding.setTo(Scalar::all(0));

	//�е� �κ��� ������ ���� �����̹����� �ȼ��� �ִ´�.
	//0~511������ �ȼ��� ������� �־��ش�.
	//�̶� �е��� ���Ե� Mat�� ���� 1~512 ���� 1~512���� �о� �־���ϹǷ� i+1, j+1
	for (int i = 0; i <= row - 1; i++) {
		for (int j = 0; j <= col - 1; j++) {
			padding.at<uchar>(i + 1, j + 1) += grayimg.at<uchar>(i, j);
		}
	}

	output.create(grayimg.size(), grayimg.type());
	output.setTo(Scalar::all(0));

	//Gaussian Filter ����
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