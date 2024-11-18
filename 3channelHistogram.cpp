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
	Mat img = imread("Lenna.png", 1);
	Mat resize_img;

	Mat r;
	Mat g;
	Mat b;

	Mat r_histo;
	Mat g_histo;
	Mat b_histo;

	resize(img, resize_img, Size(128, 128));

	int row = resize_img.rows;
	int col = resize_img.cols;

	//-------------------------
	//split RGB
	r.create(row, col, CV_8UC1);
	g.create(row, col, CV_8UC1);
	b.create(row, col, CV_8UC1);

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			r.at<uchar>(i, j) = resize_img.at<Vec3b>(i, j)[2];
		}
	}

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			g.at<uchar>(i, j) = resize_img.at<Vec3b>(i, j)[1];
		}
	}

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			b.at<uchar>(i, j) = resize_img.at<Vec3b>(i, j)[0];
		}
	}

	int r_pixel_count[256] = { 0, };
	//-------------------------
	//R
	//get image pixel
	for (int i = 0; i < col; i++) {
		for (int j = 0; j < row; j++) {
			int temp = r.at<uchar>(i, j);
			r_pixel_count[temp]++;
		}
	}

	//set histo mat size and values
	r_histo.create(270, 256, CV_8UC1);
	r_histo.setTo(Scalar::all(0));

	//input image pixel to histogram mat
	for (int i = 0; i <= 255; i++) {
		int depth = 269;
		int num = r_pixel_count[i];
		for (int j = 0; j < num; j++) {
			r_histo.at<uchar>(depth, i) = 255;
			depth--;
		}
	}
	//-------------------------
	//G
	//get image pixel
	int g_pixel_count[256] = { 0, };

	for (int i = 0; i < col; i++) {
		for (int j = 0; j < row; j++) {
			int temp = g.at<uchar>(i, j);
			g_pixel_count[temp]++;
		}
	}

	//set histo mat size and values
	g_histo.create(270, 256, CV_8UC1);
	g_histo.setTo(Scalar::all(0));

	//input image pixel to histogram mat
	for (int i = 0; i <= 255; i++) {
		int depth = 269;
		int num = g_pixel_count[i];
		for (int j = 0; j < num; j++) {
			g_histo.at<uchar>(depth, i) = 255;
			depth--;
		}
	}
	
	//-------------------------
	//B
	//get image pixel
	int b_pixel_count[256] = { 0, };

	for (int i = 0; i < col; i++) {
		for (int j = 0; j < row; j++) {
			int temp = b.at<uchar>(i, j);
			b_pixel_count[temp]++;
		}
	}

	for (int i = 0; i < 256; i++) {
		cout << i << " : " << b_pixel_count[i] << endl;
	}

	//set histo mat size and values
	b_histo.create(270, 256, CV_8UC1);
	b_histo.setTo(Scalar::all(0));

	//input image pixel to histogram mat
	for (int i = 0; i <= 256; i++) {
		int depth = 269;
		int num = b_pixel_count[i];
		for (int j = 0; j < num; j++) {
			b_histo.at<uchar>(depth, i) = 255;
			depth--;
		}
	}


	int histo_row = r_histo.rows;
	int histo_col = r_histo.cols;

	Mat merged_histo;
	merged_histo.create(histo_row, histo_col, CV_8UC3);
	merged_histo.setTo(Scalar::all(0));

	for (int i = 0; i < histo_row; i++) {
		for (int j = 0; j < histo_col; j++) {
			merged_histo.at<Vec3b>(i, j)[2] = r_histo.at<uchar>(i, j);
		}
	}

	for (int i = 0; i < histo_row; i++) {
		for (int j = 0; j < histo_col; j++) {
			merged_histo.at<Vec3b>(i, j)[1] = g_histo.at<uchar>(i, j);
		}
	}

	for (int i = 0; i < histo_row; i++) {
		for (int j = 0; j < histo_col; j++) {
			merged_histo.at<Vec3b>(i, j)[0] = b_histo.at<uchar>(i, j);
		}
	}

	imshow("merged_histo", merged_histo);

	waitKey(0);

	return 0;

}