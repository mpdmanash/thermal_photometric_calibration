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
#include<Eigen/SparseCholesky>
#include <Eigen/IterativeLinearSolvers>
#include<Eigen/SparseLU>
#include<Eigen/SparseQR>
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
typedef Eigen::SparseMatrix<double, Eigen::ColMajor> SpMat;
typedef Eigen::Triplet<double> T;
//typedef Eigen::SparseMatrix<double> SpMat;

class IRPhotoCalib{
public:
    IRPhotoCalib();
    ~IRPhotoCalib();
    void RunOptimizer(vector<int> ptx, vector<int> pty, 
                      vector<int> pptx, vector<int> ppty, 
                      vector<float> o, vector<float> op, 
                      vector<int> id, vector<int> idp, 
                      int w, int h, int div, int n_frames,
                      vector<double> &out_s, vector<double> &out_n,
                      vector<double> &out_a, vector<double> &out_b);
    void RunOptimizerCMU(vector<int> ptx, vector<int> pty, 
                      vector<int> pptx, vector<int> ppty, 
                      vector<float> o, vector<float> op, 
                      vector<int> id, vector<int> idp, 
                      int w, int h, int div, int n_frames,
                      vector<double> &out_s, vector<double> &out_n,
                      vector<double> &out_a, vector<double> &out_b);
    int RansacGains(vector<int> ptx, vector<int> pty, 
                      vector<int> pptx, vector<int> ppty, 
                      vector<float> o, vector<float> op, 
                      vector<int> id, vector<int> idp, 
                      int w, int h, int div,
                      int desired_i, int desired_ip, int offset, string dir,
                      double &out_aip, double &out_bip, vector<double> &mean_devs);
    void RunGainOptimizer(vector<int> ptx, vector<int> pty, 
                      vector<int> pptx, vector<int> ppty, 
                      vector<float> o, vector<float> op, 
                      vector<int> id, vector<int> idp, 
                      int w, int h, int div, int n_frames,
                      vector<double> &n,
                      vector<double> &out_a, vector<double> &out_b);
    void RunNewOptimizer(vector<int> ptx, vector<int> pty, 
                      vector<int> pptx, vector<int> ppty, 
                      vector<float> o, vector<float> op, 
                      vector<int> id, vector<int> idp, 
                      int w, int h, int div, int n_frames,
                      vector<double> &out_s, vector<double> &out_n,
                      vector<double> &out_a, vector<double> &out_b);
    void RunNewGainOptimizer(vector<int> ptx, vector<int> pty, 
                      vector<int> pptx, vector<int> ppty, 
                      vector<float> o, vector<float> op, 
                      vector<int> id, vector<int> idp, 
                      int w, int h, int div, int n_frames,
                      vector<double> &n,
                      vector<double> &out_a, vector<double> &out_b);
    void GetAnaJacobian(vector<int> &ptx, vector<int>& pty, 
                            vector<int>& pptx, vector<int>& ppty, 
                            vector<float>& o, vector<float>& op, 
                            vector<int>& id, vector<int>& idp, 
                            int w, int h, int div, int n_frames,
                      vector<double> n, vector<double> a, vector<double> b,
                      Eigen::MatrixXd &J);
    void RunGNAOptimizer(vector<int> ptx, vector<int> pty, 
                      vector<int> pptx, vector<int> ppty, 
                      vector<float> o, vector<float> op, 
                      vector<int> id, vector<int> idp, 
                      int w, int h, int div, int n_frames,
                      vector<double> &out_s, vector<double> &out_n,
                      vector<double> &out_a, vector<double> &out_b);
    void LinearizeSystem(vector<int> &ptx, vector<int>& pty, 
                            vector<int>& pptx, vector<int>& ppty, 
                            vector<float>& o, vector<float>& op, 
                            vector<int>& id, vector<int>& idp, 
                            int w, int h, int div, int n_frames,
                      vector<double> n, vector<double> a, vector<double> b,
                      Eigen::MatrixXd &B, SpMat &Hs, vector<int>&nz_rows);
    void GetResiduals(vector<int> &ptx, vector<int>& pty, 
                            vector<int>& pptx, vector<int>& ppty, 
                            vector<float>& o, vector<float>& op, 
                            vector<int>& id, vector<int>& idp, 
                            int w, int h, int div, int n_frames,
                      vector<double> n, vector<double> a, vector<double> b,
                      Eigen::MatrixXd &B);
     void GetSparseAnaJacobian(vector<int> &ptx, vector<int>& pty, 
                            vector<int>& pptx, vector<int>& ppty, 
                            vector<float>& o, vector<float>& op, 
                            vector<int>& id, vector<int>& idp, 
                            int w, int h, int div, int n_frames,
                      vector<double> n, vector<double> a, vector<double> b,
                      SpMat &J);
     void RunGNAOptimizerAB(vector<int> & ptx, vector<int>& pty, 
                      vector<int> &pptx, vector<int> &ppty, 
                      vector<float> &o, vector<float> &op, 
                      vector<int> &id, vector<int>&idp, 
                      int &w, int &h, int &div, int &n_frames,
                      vector<double> &out_s, vector<double> &n,
                      vector<double> &out_a, vector<double> &out_b);
    void LinearizeSystemAB(vector<int> &ptx, vector<int>& pty, 
                            vector<int>& pptx, vector<int>& ppty, 
                            vector<float>& o, vector<float>& op, 
                            vector<int>& id, vector<int>& idp, 
                            int &w, int& h, int& div, int& n_frames,
                      vector<double> &n, vector<double>& a, vector<double>& b,
                      Eigen::MatrixXd &B, SpMat &Hs);
    void GetResidualsAB(vector<int> &ptx, vector<int>& pty, 
                            vector<int>& pptx, vector<int>& ppty, 
                            vector<float>& o, vector<float>& op, 
                            vector<int>& id, vector<int>& idp, 
                            int &w, int& h, int &div, int& n_frames,
                      vector<double>& n, vector<double>& a, vector<double>& b,
                      Eigen::MatrixXd &B);
     void GetSparseAnaJacobianAB(vector<int> &ptx, vector<int>& pty, 
                            vector<int>& pptx, vector<int>& ppty, 
                            vector<float>& o, vector<float>& op, 
                            vector<int>& id, vector<int>& idp, 
                            int& w, int& h, int &div, int &n_frames,
                      vector<double> &n, vector<double>&a, vector<double> &b,
                      SpMat &J);
    bool isInsideFrame(int x, int y, int w, int h);
    float getKernelO(cv::Mat &im, int x, int y, int k);
    float getCorrected(float o, int x, int y, vector<double> &a, vector<double> &b, vector<double> &s, vector<double> &n, int fid, int w, int h, int div, Mat &imNis, double minn, double maxn);
    float getGainCorrected(float o, int x, int y, vector<double> &a, vector<double> &b, vector<double> &s, vector<double> &n, int fid, int w, int h, int div);
    void getStartPoints(vector<int> ptx, vector<int> pty, 
                        vector<int> pptx, vector<int> ppty,
                        int w, int h, int div);
    void getRelativeGains(double a1, double b1, double a2, double b2, double & a12, double & b12);
    int SerializeLocation(int x, int y, int w, int h);
    void DeSerializeLocation(int id, int w, int h, int & x, int & y);
    void EstimateSpatialParameters(vector<int> &ptx, vector<int>& pty, 
                                   vector<int>& pptx, vector<int>& ppty, 
                                   vector<float>& o, vector<float>& op, 
                                   vector<int>& id, vector<int>& idp,
                                   vector<double> & all_aips, vector<double> & all_bips,
                                   int w, int h, int div,
                                   string PS_correspondance, string PS_location, string A_location);
     void exportCorrespondences(vector<int> &ptx, vector<int>& pty, 
                                   vector<int>& pptx, vector<int>& ppty, 
                                   vector<float>& o, vector<float>& op, 
                                   vector<int>& id, vector<int>& idp,
                                   string filename);
     void exportKFPT(vector<double> & all_aips, vector<double> & all_bips, string filename);
  double e_photo_error;
};

#endif
