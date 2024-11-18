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
#define SQR(x) ((x)*(x))

void compute_coefs3(double c[5], double sigma) {
        /*
        *  "Recursive Implementation of the gaussian filter.",
        *   Ian T. Young , Lucas J. Van Vliet, Signal Processing 44, Elsevier 1995.
        */
    double q = 0;
    if (sigma >= 2.5)
        q = 0.98711 * sigma - 0.96330;
    else if ((sigma >= 0.5) && (sigma < 2.5))
        q = 3.97156 - 4.14554 * (double)sqrt((double)1 - 0.26891 * sigma);
    else
        q = 0.1147705018520355224609375;

    double q2 = q * q;
    double q3 = q * q2;
    c[0] = (1.57825 + (2.44413 * q) + (1.4281 * q2) + (0.422205 * q3));
    c[1] = ((2.44413 * q) + (2.85619 * q2) + (1.26661 * q3));
    c[2] = (-((1.4281 * q2) + (1.26661 * q3)));
    c[3] = ((0.422205 * q3));
    c[4] = 1.0 - (c[1] + c[2] + c[3]) / c[0];
}

void retinex_scale_distribution(const int nscales/*=3*/, const int s/*=240*/,
    double scales[]) {
        //  ASSERT(nscales>=3);
    double size_step = (double)s / (double)nscales;
    for (int i = 0; i < nscales; ++i)
        scales[i] = 2. + (double)i * size_step;
}

void gausss_mooth(double* in, int size, int rowstride, double* out, double b[5]) {
        /* forward pass */
    int bufsize = size + 3;
    size -= 1;
    double* w1 = new double[bufsize];
    double* w2 = new double[bufsize];
    memset(w1, 0, sizeof(double) * bufsize);
    memset(w2, 0, sizeof(double) * bufsize);
    w1[0] = in[0];
    w1[1] = in[0];
    w1[2] = in[0];
    for (int i = 0, n = 3; i <= size; i++, n++) {
        w1[n] = (double)(b[4] * in[i * rowstride] + ((b[1] * w1[n - 1] + b[2] * w1[n - 2] + b[3] * w1[n - 3]) / b[0]));
    }
    /* backward pass */
    w2[size + 1] = w1[size + 3];
    w2[size + 2] = w1[size + 3];
    w2[size + 3] = w1[size + 3];
    for (int i = size, n = i; i >= 0; i--, n--) {
        w2[n] = out[i * rowstride] = (double)(b[4] * w1[n] + ((b[1] * w2[n + 1] + b[2] * w2[n + 2] + b[3] * w2[n + 3]) / b[0]));
    }
    delete[] w1;
    delete[] w2;
}


void image_statistics(double* img, int size/*=width*height*/,
    double* mean, double* std) {
    double s = 0, ss = 0;
    for (int i = 0; i < size; i++) {
        double a = img[i];
        s += a;
        ss += SQR(a);
    };
    *mean = s / size;
    *std = sqrt((ss - s * s / size) / size);
}

void rescale_range(double* data, int size) {
    double mean, sig;
    image_statistics(&data[0], size, &mean, &sig);

    double max_val = mean + 1.2 * sig;
    double min_val = mean - 1.2 * sig;
    double range = max_val - min_val;
    if (!range) range = 1.0;
    range = 255. / range;
    // change the range;
    for (int i = 0; i < size; i++) {
        data[i] = (data[i] - min_val) * range;
    }
}

void retinex_process(double* src, int width, int height, double* dst) {
    const int nfilter = 3;
    double sigma[nfilter];
    double c[5];
    int default_scale = 240;
    retinex_scale_distribution(nfilter, default_scale, sigma);
    int csize = width * height;
    double* in = new double[csize];
    double* out = new double[csize];
    memset(dst, 0, csize * sizeof(double));
    // scale-space gauss_smooth;
    for (int i = 0; i < nfilter; i++) {
        compute_coefs3(c, sigma[i]);
        // copy src to temp. buffer(=in);
        for (int pos = 0; pos < csize; pos++)
            in[pos] = double(src[pos] + 1.0);
        // (1) horizontal convolution(stride = 1 for grey);
        for (int y = 0; y < height; y++)
            gausss_mooth(&in[y * width], width, 1, &out[y * width], c);
        // (2) vertical convolution(stride = height for grey);
        memcpy(in, out, csize * sizeof(double));
        for (int x = 0; x < width; x++)
            gausss_mooth(&in[x], height, width, &out[x], c);
        // 각 스케일에서 반사 성분을 누적;
        for (int pos = 0; pos < csize; pos++)
            dst[pos] += log(src[pos] + 1) - log(out[pos]);
    }
    // scale to [0,255];
    rescale_range(&dst[0], csize);
    delete[] in;
    delete[] out;
};

void main() {
    int width = 580;
    int height = 420;
    double dst[420][580] = { 0, };
    double src[420][580] = { 0, };

    Mat img = imread("0124_REAL_L_1.bmp", 0);

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            src[i][j] == img.at<uchar>(i, j);
        }
    }

    retinex_process(*src, width, height, *dst);
    Mat output(height, width, CV_8UC1, Scalar(0));

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            output.at<uchar>(i, j) = dst[i][j];
        }
    }
}