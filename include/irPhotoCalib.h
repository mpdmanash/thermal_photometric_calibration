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

struct PTAB{
  double a;
  double b;
};

class IRPhotoCalib{
public:
    IRPhotoCalib(bool useKeyframes=true);
    ~IRPhotoCalib();
    PTAB ProcessCurrentFrame(vector<vector<float> > intensity_history,
                             vector<vector<float> > intensity_current, 
                             vector<int> frame_ids_history, bool thisKF);
    int EstimateGainsRansac(vector<float> oi, vector<float> oip,
                            double &out_aip, double &out_bip);
    float getCorrected(float o, int x, int y, vector<double> &a, vector<double> &b, vector<double> &s, vector<double> &n, int fid, int w, int h, int div, Mat &imNis, double minn, double maxn);
    float getGainCorrected(float o, int x, int y, vector<double> &a, vector<double> &b, vector<double> &s, vector<double> &n, int fid, int w, int h, int div);
    void getRelativeGains(double a1, double b1, double a2, double b2, double & a12, double & b12);
    void chainGains(double a01, double b01, double a12, double b12, double & a02, double & b02);
    double e_photo_error;
private:
    bool m_useKeyframes;
    vector<PTAB> m_params_PT;
    vector<int> m_KF_ids;
    int m_latest_KF_id;
    int m_frame_id;
    double m_epsilon_gap, m_epsilon_base;
    PTAB getPrevAB();
};

#endif
