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
#include <math.h>

using namespace std;
using namespace cv;

// Config sets
const int k_frame_history = 10;
int k_div = 16;
float k_SP_threshold = 0.97;
bool k_calibrate_SP = true;
int k_max_corres_per_frame = 300; //600 in general, but lesser 20 or 30 for spatial
int k_max_corres_per_history = 30;
int k_min_corres_per_history = 4;

void ReadCorrespondences(string csv_filename, 
                         vector<vector<vector<float> > > & all_intensity_history,
                         vector<vector<vector<float> > > & all_intensity_current,
                         vector<vector<int> > & all_history_frame_diffs,
                         vector<vector<vector<pair<int,int> > > > & all_pixels_history,
                         vector<vector<vector<pair<int,int> > > > & all_pixels_current,
                         vector<int> & all_isKFs){
    all_intensity_history.clear(); all_intensity_current.clear(); all_history_frame_diffs.clear(); all_pixels_history.clear(); all_pixels_current.clear();
    ifstream c_file (csv_filename); string value;
    int prev_CFID = 1; int columns = 10; int prev_HKFID = -1;
    vector<vector<float> > CFID_intensity_history, CFID_intensity_current;
    vector<vector<pair<int,int> > > CFID_pixels_history, CFID_pixels_current;
    vector<int> CFID_history_frame_diffs;
    vector<float> CFID_HKFID_intensity_history, CFID_HKFID_intensity_current;
    vector<pair<int,int> > CFID_HKFID_pixels_history, CFID_HKFID_pixels_current;
    bool done_all_lines = false;
    int x_history = 0; int x_current = 0; int CFID_is_KF = 0;
    while ( c_file.good() ){
        for(int i=0; i<columns-1; i++){
            getline ( c_file, value, ',' );
            if (value.empty()) {done_all_lines=true; break;}
            switch(i) {
                case 0 : // We are reading the current frame ID
                    {
                        int CFID = stoi(value);
                        if (CFID!=prev_CFID){ // We have moved onto the next CFID
                            CFID_intensity_history.push_back(CFID_HKFID_intensity_history);
                            CFID_intensity_current.push_back(CFID_HKFID_intensity_current);
                            CFID_history_frame_diffs.push_back(prev_HKFID);
                            CFID_pixels_history.push_back(CFID_HKFID_pixels_history);
                            CFID_pixels_current.push_back(CFID_HKFID_pixels_current);
                            CFID_HKFID_intensity_history.clear(); CFID_HKFID_intensity_current.clear();
                            CFID_HKFID_pixels_history.clear(); CFID_HKFID_pixels_current.clear();
                            all_intensity_history.push_back(CFID_intensity_history);
                            all_intensity_current.push_back(CFID_intensity_current);
                            all_history_frame_diffs.push_back(CFID_history_frame_diffs);
                            all_pixels_history.push_back(CFID_pixels_history);
                            all_pixels_current.push_back(CFID_pixels_current);
                            all_isKFs.push_back(CFID_is_KF);
                            CFID_intensity_history.clear(); CFID_intensity_current.clear(); CFID_history_frame_diffs.clear();
                            CFID_pixels_history.clear(); CFID_pixels_current.clear();
                            prev_HKFID=-1;
                        }
                        prev_CFID = CFID;
                    }
                    break;
                case 1 :
                    {
                        int HKFID = stoi(value);
                        if (prev_HKFID!=HKFID && prev_HKFID!=-1){
                            CFID_intensity_history.push_back(CFID_HKFID_intensity_history);
                            CFID_intensity_current.push_back(CFID_HKFID_intensity_current);
                            CFID_history_frame_diffs.push_back(prev_HKFID);
                            CFID_pixels_history.push_back(CFID_HKFID_pixels_history);
                            CFID_pixels_current.push_back(CFID_HKFID_pixels_current);
                            CFID_HKFID_intensity_history.clear(); CFID_HKFID_intensity_current.clear();
                            CFID_HKFID_pixels_history.clear(); CFID_HKFID_pixels_current.clear();
                            CFID_history_frame_diffs.push_back(HKFID);
                        }
                        prev_HKFID = HKFID;
                    }
                    break;
                case 2 :
                    {
                        float o_history = stof(value);
                        CFID_HKFID_intensity_history.push_back(o_history);
                    }
                    break;
                case 3 :
                    {
                        float o_current = stof(value);
                        CFID_HKFID_intensity_current.push_back(o_current);
                    }
                    break;
                case 4 : {CFID_is_KF = stoi(value);break;}
                case 5 : {x_history = (int)round(stof(value)); break;}
                case 6 :
                    {
                        int y_history = (int)round(stof(value));
                        CFID_HKFID_pixels_history.push_back(make_pair(x_history, y_history));
                    }
                    break;
                case 7 : {x_current = (int)round(stof(value)); break;}
                case 8 :
                    {
                        int y_current = (int)round(stof(value));
                        CFID_HKFID_pixels_current.push_back(make_pair(x_current, y_current));
                    }
                    break;
            }
        }
        if (!done_all_lines) getline ( c_file, value);
        else{
            CFID_intensity_history.push_back(CFID_HKFID_intensity_history);
            CFID_intensity_current.push_back(CFID_HKFID_intensity_current);
            CFID_history_frame_diffs.push_back(prev_HKFID);
            CFID_pixels_history.push_back(CFID_HKFID_pixels_history);
            CFID_pixels_current.push_back(CFID_HKFID_pixels_current);
            all_intensity_history.push_back(CFID_intensity_history);
            all_intensity_current.push_back(CFID_intensity_current);
            all_history_frame_diffs.push_back(CFID_history_frame_diffs);
            all_pixels_history.push_back(CFID_pixels_history);
            all_pixels_current.push_back(CFID_pixels_current);
            all_isKFs.push_back(CFID_is_KF);
        }
    }
}

int main(int argc, char **argv){
    cv::String keys =
        "{@input_path |<none>           | Path to the video file}"         
        "{@correspondence_path  |/path/to/file.xml| Path to the correspondence file}"
        "{help   |      | show help message}";      // optional, show help optional
    CommandLineParser parser(argc, argv, keys);
    parser.printMessage();
    string input_path = parser.get<string>(0);
    string correspondence_path = parser.get<string>(1);
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
    int frame_counter = 0;
    Mat prev_frame, frame_3, frame, corrected_frame, res;

    // ===== Read Correspondences
    vector<vector<vector<float> > > all_intensity_history, all_intensity_current;
    vector<vector<int> > all_history_frame_diffs; vector<int> all_isKFs;
    vector<vector<vector<pair<int,int> > > > all_pixels_history, all_pixels_current;
    ReadCorrespondences(correspondence_path, all_intensity_history, all_intensity_current, all_history_frame_diffs, all_pixels_history, all_pixels_current, all_isKFs);

    // ===== Setup for IRPC =================================================
    IRPhotoCalib * calib;
    vector<int> ptx, pty, pptx, ppty, id, idp; 
    vector<float> o, op;
    while(video_in.isOpened()){
        video_in >> frame_3;
        if(frame_3.empty()) continue;
        cvtColor(frame_3, frame, CV_BGR2GRAY);        
        if (frame_counter>=1){
            PTAB current_params = calib->ProcessCurrentFrame(all_intensity_history[frame_counter-1], all_intensity_current[frame_counter-1], all_history_frame_diffs[frame_counter-1],
                                                             all_pixels_history[frame_counter-1], all_pixels_current[frame_counter-1], (bool)all_isKFs[frame_counter-1]);
            corrected_frame = calib->getCorrectedImage(frame, current_params);
        }
        else {corrected_frame = frame.clone(); calib = new IRPhotoCalib(frame.cols,frame.rows,k_div,k_calibrate_SP,k_SP_threshold,true);}
        hconcat(frame, corrected_frame, res);
        imshow(video_name, res);
        waitKey(1);
        frame_counter++;
    }
}