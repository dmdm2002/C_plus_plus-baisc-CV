#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio.hpp>

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
	Mat img = imread("Lenna.png", 1);
	Mat r;
	Mat g;
	Mat b;

	int row = img.rows;
	int col = img.cols;


	//-------------------------
	//split RGB
	r.create(row, col, CV_8UC1);
	g.create(row, col, CV_8UC1);
	b.create(row, col, CV_8UC1);


	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			r.at<uchar>(i, j) = img.at<Vec3b>(i, j)[2];
		}
	}

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			g.at<uchar>(i, j) = img.at<Vec3b>(i, j)[1];
		}
	}

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			b.at<uchar>(i, j) = img.at<Vec3b>(i, j)[0];
		}
	}

	//-------------------------
	//padding
	Mat r_padding;
	Mat g_padding;
	Mat b_padding;

	r_padding.create(row + 2, col + 2, CV_8UC1);
	r_padding.setTo(Scalar::all(0));

	g_padding.create(row + 2, col + 2, CV_8UC1);
	g_padding.setTo(Scalar::all(0));

	b_padding.create(row + 2, col + 2, CV_8UC1);
	b_padding.setTo(Scalar::all(0));

	for (int i = 0; i <= row - 1; i++) {
		for (int j = 0; j <= col - 1; j++) {
			r_padding.at<uchar>(i + 1, j + 1) += r.at<uchar>(i, j);
		}
	}

	for (int i = 0; i <= row - 1; i++) {
		for (int j = 0; j <= col - 1; j++) {
			g_padding.at<uchar>(i + 1, j + 1) += g.at<uchar>(i, j);
		}
	}

	for (int i = 0; i <= row - 1; i++) {
		for (int j = 0; j <= col - 1; j++) {
			b_padding.at<uchar>(i + 1, j + 1) += b.at<uchar>(i, j);
		}
	}

	//-------------------------
	//Gaussian Filter
	double GFilter[3][3];
	GaussianFilter(GFilter);

	Mat r_output;
	Mat g_output;
	Mat b_output;
	r_output.create(row, col, CV_8UC1);
	r_output.setTo(Scalar::all(0));

	g_output.create(row, col, CV_8UC1);
	g_output.setTo(Scalar::all(0));

	b_output.create(row, col, CV_8UC1);
	b_output.setTo(Scalar::all(0));

	for (int i = 0; i <= row - 1; i++) {
		for (int j = 0; j <= col - 1; j++) {
			for (int k = 0; k <= 2; k++) {
				for (int l = 0; l <= 2; l++) {
					r_output.at<uchar>(i, j) += r_padding.at<uchar>(i + k, j + l) * GFilter[k][l];
				}
			}
		}
	}

	for (int i = 0; i <= row - 1; i++) {
		for (int j = 0; j <= col - 1; j++) {
			for (int k = 0; k <= 2; k++) {
				for (int l = 0; l <= 2; l++) {
					g_output.at<uchar>(i, j) += g_padding.at<uchar>(i + k, j + l) * GFilter[k][l];
				}
			}
		}
	}

	for (int i = 0; i <= row - 1; i++) {
		for (int j = 0; j <= col - 1; j++) {
			for (int k = 0; k <= 2; k++) {
				for (int l = 0; l <= 2; l++) {
					b_output.at<uchar>(i, j) += b_padding.at<uchar>(i + k, j + l) * GFilter[k][l];
				}
			}
		}
	}


	//-------------------------
	//Merge RGB channels
	Mat merged_image;
	merged_image.create(row, col, CV_8UC3);
	merged_image.setTo(Scalar::all(0));

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			merged_image.at<Vec3b>(i, j)[2] = r_output.at<uchar>(i, j);
		}
	}

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			merged_image.at<Vec3b>(i, j)[1] = g_output.at<uchar>(i, j);
		}
	}

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			merged_image.at<Vec3b>(i, j)[0] = b_output.at<uchar>(i, j);
		}
	}

	imshow("origin", img);
	imshow("merged_image", merged_image);

	waitKey(0);

	return 0;
}