#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <iostream>
#include <stdio.h>
#include <cmath>
#include <algorithm>

using namespace cv;
using namespace std;

int main() {
	Mat img = imread("coin.png", COLOR_BGR2GRAY);
	Mat resize_img;
	Mat histo;

	resize(img, resize_img, Size(128, 128));

	int row = resize_img.rows;
	int col = resize_img.cols;

	//get image pixel
	int pixel_count[256] = { 0, };
	for (int i = 0; i < col; i++) {
		for (int j = 0; j < row; j++) {
			int temp = resize_img.at<uchar>(i, j);
			pixel_count[temp]++;
		}
	}
	float scale_factor = 255.0f / (float)(row * col);

	//Equalization
	int pixel_stack = 0;
	for (int i = 0; i < 256; i++) {
		pixel_stack += pixel_count[i];
		pixel_count[i] = (int)((scale_factor * pixel_stack) + 0.5);
	}

	//make Equlization image
	Mat output;
	output.create(row, col, CV_8UC1);

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			int temp = resize_img.at<uchar>(i, j);
			output.at<uchar>(i, j) = pixel_count[temp];
		}
	}

	//get equalization image pixel
	int equl_pixel_count[256] = { 0, };
	for (int i = 0; i < col; i++) {
		for (int j = 0; j < row; j++) {
			int temp = output.at<uchar>(i, j);
			equl_pixel_count[temp]++;
		}
	}

	//set histo mat size and values
	histo.create(1000, 256, output.type());
	histo.setTo(Scalar::all(0));

	//input image pixel to histogram mat
	for (int i = 0; i < 256; i++) {
		int depth = 999;
		int num = equl_pixel_count[i];
		for (int j = 0; j < num; j++) {
			histo.at<uchar>(depth, i) = 255;
			depth--;
		}
	}

	resize(output, output, Size(256, 256));

	imshow("img", output);
	imshow("histo", histo);
	imshow("test", img);

	waitKey(0);

	return 0;

}