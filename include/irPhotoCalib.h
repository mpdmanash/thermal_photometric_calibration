/*
 * Author: Manash Pratim Das (mpdmanash@iitkgp.ac.in) 
 */

#ifndef IRPHOTOCALIB_H
#define IRPHOTOCALIB_H

#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <vector>
#include <omp.h>
#include <iomanip>
#include <cstdio>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <Eigen/SparseCholesky>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SparseLU>
#include <Eigen/SparseQR>
//#include<Eigen/SPQRSupport>
#include <Eigen/OrderingMethods>
#include <time.h>
#include <sys/time.h>
#include <random>
#include <algorithm>
#include <iterator>
#include <string>


//#define M_PI 3.14159265
using namespace std;
using namespace cv;

class IRPhotoCalib{
public:
    IRPhotoCalib();
    ~IRPhotoCalib();
    int EstimateGainsRansac(vector<float> oi, vector<float> oip,
                            double &out_aip, double &out_bip);
    float getCorrected(float o, int x, int y, vector<double> &a, vector<double> &b, vector<double> &s, vector<double> &n, int fid, int w, int h, int div, Mat &imNis, double minn, double maxn);
    float getGainCorrected(float o, int x, int y, vector<double> &a, vector<double> &b, vector<double> &s, vector<double> &n, int fid, int w, int h, int div);
    void getRelativeGains(double a1, double b1, double a2, double b2, double & a12, double & b12);
  double e_photo_error;
};

#endif
