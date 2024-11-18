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

int main() {
	Mat img = imread("Lenna.png", IMREAD_GRAYSCALE);
	Mat resize_img;
	Mat histo;

	//image resize
	resize(img, resize_img, Size(128, 128));

	int row = resize_img.rows;
	int col = resize_img.cols;

	int pixel_count[256] = { 0, };

	//get image pixel
	int sumPixels = 0;
	for (int i = 0; i < col; i++) {
		for (int j = 0; j < row; j++) {
			int temp = resize_img.at<uchar>(i, j);
			pixel_count[temp]++;
		}
	}

	//set histo mat size and values
	histo.create(256, 256, resize_img.type());
	histo.setTo(Scalar::all(0));

	//input image pixel to histogram mat
	for (int i = 0; i <= 255; i++) {
		int depth = 255;
		int num = pixel_count[i];
		for (int j = 0; j < num; j++) {
			histo.at<uchar>(depth, i) = 255;
			depth--;
		}
	}
	//---------------------------------------------------
	//Gaussian Blur Histogram
	Mat img_blur;

	GaussianBlur(resize_img, img_blur, Size(7, 7), 0);

	Mat resize_blur_img;
	Mat blur_histo;

	int blur_pixel_count[256] = { 0, };

	//get blur image pixel
	for (int i = 0; i < col; i++) {
		for (int j = 0; j < row; j++) {
			int temp = img_blur.at<uchar>(i, j);
			blur_pixel_count[temp]++;
		}
	}

	//set blur histo mat size and values
	blur_histo.create(256, 256, img_blur.type());
	blur_histo.setTo(Scalar::all(0));


	//input blur image pixel to blur histogram mat
	for (int i = 0; i <= 255; i++) {
		int depth = 255;
		int num = blur_pixel_count[i];
		for (int j = 0; j < num; j++) {
			blur_histo.at<uchar>(depth, i) = 255;
			depth--;
		}
	}
	
	imshow("histo_blur", blur_histo);
	imshow("histo", histo);
	imshow("img", img);
	imshow("img_blur", img_blur);

	waitKey(0);

	return 0;
}