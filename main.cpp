#include <opencv2/features2d.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>      //for imshow
#include <opencv2/core.hpp>
#include <vector>
#include <queue>
#include <iostream>
#include <iomanip>

#include "utils.h"
#include "irPhotoCalib.h"
#include<cstdlib>
#include <time.h>
#include <dirent.h>
#include <string>
#include <stdio.h>
#include <fstream>
#include <opencv2/opencv.hpp>

#include <random>
#include <algorithm>
#include <iterator>

using namespace std;
using namespace cv;

// Config sets
const int k_o_ws = 32;
const double k_akaze_thresh = 3e-4; // AKAZE detection threshold set to locate about 1000 keypoints
const double k_ransac_thresh = 2.5f; // RANSAC inlier threshold
const double k_nn_match_ratio = 0.8f; // Nearest-neighbour matching ratio
const int k_frame_history = 10;
int k_div = 8;
int k_max_corres_per_frame = 200; //600 in general, but lesser 20 or 30 for spatial
int k_max_corres_per_history = 100;
int k_min_corres_per_history = 4;
double k_epsilon_gap = 0.01;
double k_epsilon_base = 0.01;

struct PTAB{
  double a;
  double b;
};

void getRelativeGains(double a1, double b1, double a2, double b2, double & a12, double & b12){
  double e12 = (a2-b2)/(a1-b1);
  b12 = (b2-b1)/(a1-b1);
  a12 = e12 + b12;
}

void chainGains(double a01, double b01, double a12, double b12, double & a02, double & b02){
  double e02 = (a01-b01) * (a12 - b12);
  b02 = b01+(a01-b01)*b12;
  a02 = e02 + b02;
}


int main(int argc, char **argv){
    CommandLineParser parser(argc, argv, "{@input_path |0|input path can be a camera id, like 0,1,2 or a video filename}");
    parser.printMessage();
    string input_path = parser.get<string>(0);
    string video_name = input_path;
    namedWindow(video_name, WINDOW_NORMAL);
    VideoCapture video_in;

    if ( ( isdigit(input_path[0]) && input_path.size() == 1 ) )
    {
    int camera_no = input_path[0] - '0';
        video_in.open( camera_no );
    }
    else {
        video_in.open(video_name);
    }

    if(!video_in.isOpened()) {
        cerr << "Couldn't open " << video_name << endl;
        return 1;
    }
    Ptr<AKAZE> feature_descriptor = AKAZE::create();
    feature_descriptor->setThreshold(k_akaze_thresh);
    // Ptr<ORB> feature_descriptor = ORB::create();
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

    int frame_counter = 0;
    Mat full_frame, prev_frame, frame_3, frame;

    // DS that stores history
    vector<Mat> prev_descs;
    vector<vector<KeyPoint> > prev_kps;
    vector<vector<float> > prev_intensities;
    vector<PTAB> prev_PTs;

    // ===== Setup for IRPC =================================================
    bool estimate_PT = true;
    bool estimate_PS = true;
    bool no_drift = false;
    IRPhotoCalib calib;
    calib.e_photo_error = calib.e_photo_error/10.0;
    vector<int> ptx, pty, pptx, ppty, id, idp; 
    vector<float> o, op;
    int w=320; int h=240;

    while(video_in.isOpened()){
        video_in >> full_frame;
        if(full_frame.empty()) continue;
        resize(full_frame,frame_3,cv::Size(w,h));
        cvtColor(frame_3, frame, CV_BGR2GRAY);

        if (frame_counter >= 1){ 
            TickMeter tm;
            tm.start();

            // Compute This Frame Properties
            vector<KeyPoint> kp; Mat desc; vector<float> intensity;
            feature_descriptor->detectAndCompute(frame, noArray(), kp, desc);
            GetFrameIntensities(kp,frame,k_o_ws,intensity);

            // Match this frame with previous frames
            float i_ratio = 0;
            int total_num_corres = 0;
            int max_prev_frame = std::min(k_frame_history, (int)prev_kps.size());
            double a_origin_previous = prev_PTs.back().a; double b_origin_previous = prev_PTs.back().b;
            double w_a = 0; double w_b = 0; int w_count = 0;
            for(int i=max_prev_frame-1; i>=0 && total_num_corres<k_max_corres_per_frame; i--){ // Do for loop until you hit max history available or allowed or if you hit total_bum_corres
                vector<int> prev_inliers, inliers;
                i_ratio = GetInliers(prev_descs[i],prev_kps[i],desc,kp,matcher,k_ransac_thresh,k_nn_match_ratio,prev_inliers,inliers);
                int this_num_corres = std::min((int)prev_inliers.size(), k_max_corres_per_history);
                this_num_corres = (this_num_corres<k_min_corres_per_history)?0:this_num_corres;
                if (this_num_corres==0) continue;

                // Prepare stuff for the Ransac Gain Estimation
                vector<float>intensity_prev, intensity_current;
                for(int j=0; j<this_num_corres; j++){
                    intensity_prev.push_back(prev_intensities[i][prev_inliers[j]]);
                    intensity_current.push_back(intensity[inliers[j]]);
                }
                double a_history_current, b_history_current, a_origin_current, b_origin_current, a_previous_current, b_previous_current;
                int support_points = calib.EstimateGainsRansac(intensity_prev, intensity_current, a_history_current, b_history_current);
                double a_origin_history = prev_PTs[i].a; double b_origin_history = prev_PTs[i].b;
                chainGains(a_origin_history, b_origin_history, a_history_current, b_history_current, a_origin_current, b_origin_current);
                getRelativeGains(a_origin_previous, b_origin_previous, a_origin_current, b_origin_current, a_previous_current, b_previous_current);
                w_a += a_previous_current*support_points; w_b += b_previous_current*support_points; w_count += support_points;

                total_num_corres += this_num_corres;
            }
            tm.stop();
            float fps = 1. / tm.getTimeSec();

            double w_a_previous_current = w_a/w_count; double w_b_previous_current = w_b/w_count;
            
            // Drift adjustment
            double delta = (1.0 - (w_a_previous_current-w_b_previous_current)) * k_epsilon_gap;
            w_a_previous_current = w_a_previous_current -(w_a_previous_current-1)*k_epsilon_base + k_epsilon_gap;
            w_b_previous_current = w_b_previous_current -(w_b_previous_current)*k_epsilon_base - k_epsilon_gap;

            double a_origin_current, b_origin_current;
            chainGains(a_origin_previous, b_origin_previous, w_a_previous_current, w_b_previous_current, a_origin_current, b_origin_current);

            cout << "FPS:" << fps << " Inlier Ratio:" << i_ratio << " max C:" << total_num_corres << "|| PT A:" << w_a_previous_current << " PT B:" << w_b_previous_current << endl;

            // Saving this frame info for the next frames
            prev_frame = frame;
            prev_kps.push_back(kp);
            prev_descs.push_back(desc);
            prev_intensities.push_back(intensity);
            prev_PTs.push_back({a_origin_current, b_origin_current});
            if(prev_kps.size()>k_frame_history){
                prev_kps.erase(prev_kps.begin());
                prev_descs.erase(prev_descs.begin());
                prev_intensities.erase(prev_intensities.begin());
                prev_PTs.erase(prev_PTs.begin());
            }
        }
        else{
            vector<KeyPoint> kp; Mat desc; vector<float> intensity;
            feature_descriptor->detectAndCompute(frame, noArray(), kp, desc);
            GetFrameIntensities(kp,frame,k_o_ws,intensity);
            prev_kps.push_back(kp);
            prev_descs.push_back(desc);
            prev_intensities.push_back(intensity);
            prev_frame = frame;
            prev_PTs.push_back({1.0,0.0});
        }
        frame_counter++;
    }
}