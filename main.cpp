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
const int k_o_ws = 8;
const double k_akaze_thresh = 3e-8; // AKAZE detection threshold set to locate about 1000 keypoints
const double k_ransac_thresh = 4.5f; // RANSAC inlier threshold
const double k_nn_match_ratio = 0.96f; // Nearest-neighbour matching ratio
const int k_frame_history = 10;
int k_div = 8;
int k_max_corres_per_frame = 300; //600 in general, but lesser 20 or 30 for spatial
int k_max_corres_per_history = 30;
int k_min_corres_per_history = 4;


int main(int argc, char **argv){
    CommandLineParser parser(argc, argv, "{@input_path |0|input path can be a camera id, like 0,1,2 or a video filename}");
    parser.printMessage();
    string input_path = "/media/kimsk/DATA/Manash_Data/IRPhotoCalib_dataset/AM09_LWIR_V000.avi"; //parser.get<string>(0);
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
    vector<int> frame_ids;

    // ===== Setup for IRPC =================================================
    IRPhotoCalib calib(true);
    vector<int> ptx, pty, pptx, ppty, id, idp; 
    vector<float> o, op;
    int w=320; int h=240;
    while(video_in.isOpened()){
        video_in >> full_frame;
        if(full_frame.empty()) continue;
        resize(full_frame,frame_3,cv::Size(w,h));
        cvtColor(frame_3, frame, CV_BGR2GRAY);

        if (frame_counter >= 1){ 
            // Compute This Frame Properties
            vector<KeyPoint> kp; Mat desc; vector<float> intensity;
            feature_descriptor->detectAndCompute(frame, noArray(), kp, desc);
            GetFrameIntensities(kp,frame,k_o_ws,intensity);

            // Match this frame with previous frames
            float i_ratio = 0;
            int total_num_corres = 0;
            vector<vector<float> > all_intensity_history, all_intensity_current; vector<int> all_frame_ids_history;
            for(int i=frame_ids.size()-1; i>=0; i--){ // loop through the keyframes
                vector<int> prev_inliers, inliers;
                i_ratio = GetInliers(prev_descs[i],prev_kps[i],desc,kp,matcher,k_ransac_thresh,k_nn_match_ratio,prev_inliers,inliers);
                int this_num_corres = std::min((int)prev_inliers.size(), k_max_corres_per_history);
                this_num_corres = (this_num_corres<k_min_corres_per_history)?0:this_num_corres;
                if (this_num_corres==0) continue;

                vector<float>intensity_prev, intensity_current;
                for(int j=0; j<this_num_corres; j++){
                    intensity_prev.push_back(prev_intensities[i][prev_inliers[j]]);
                    intensity_current.push_back(intensity[inliers[j]]);
                }
                all_intensity_history.push_back(intensity_prev);
                all_intensity_current.push_back(intensity_current);
                all_frame_ids_history.push_back(frame_counter-frame_ids[i]);
            }
            double random_number = (double) rand() / (RAND_MAX);
            cout  << "Frame Number=" << frame_counter << " random number=" << random_number << endl;
            bool thisKF = (random_number < 1./30.)? true:false;
            PTAB current_params = calib.ProcessCurrentFrame(all_intensity_history, all_intensity_current, all_frame_ids_history, thisKF);
            if (thisKF)
            {
                prev_kps.push_back(kp);
                prev_descs.push_back(desc);
                prev_intensities.push_back(intensity);
                frame_ids.push_back(frame_counter);
                cout << "Adding a keyframe at frame " << frame_counter << endl;
            }
            

            Mat corrected_frame = frame * (current_params.a-current_params.b) + current_params.b;
            Mat res;
            hconcat(frame, corrected_frame, res);

            imshow(video_name, res);
            waitKey(1);

            //cout << "FPS:" << fps << " Inlier Ratio:" << i_ratio << " max C:" << w_count << "|| PT A:" << a_origin_current << " PT B:" << b_origin_current << endl;

            // Saving this frame info for the next frames
            prev_frame = frame;
            if(prev_kps.size()>k_frame_history){
                prev_kps.erase(prev_kps.begin());
                prev_descs.erase(prev_descs.begin());
                prev_intensities.erase(prev_intensities.begin());
                frame_ids.erase(frame_ids.begin());
            }
        }
        else{
            vector<KeyPoint> kp; Mat desc; vector<float> intensity;
            feature_descriptor->detectAndCompute(frame, noArray(), kp, desc);
            GetFrameIntensities(kp,frame,k_o_ws,intensity);
            prev_kps.push_back(kp);
            prev_descs.push_back(desc);
            prev_intensities.push_back(intensity);
            frame_ids.push_back(frame_counter);
            prev_frame = frame;
        }
        frame_counter++;
    }
}