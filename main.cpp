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
const int k_frame_history = 10;
int k_div = 8;
int k_max_corres_per_frame = 300; //600 in general, but lesser 20 or 30 for spatial
int k_max_corres_per_history = 30;
int k_min_corres_per_history = 4;

void ReadCorrespondences(string csv_filename, 
                         vector<vector<vector<float> > > & all_intensity_history,
                         vector<vector<vector<float> > > & all_intensity_current,
                         vector<vector<int> > & all_history_frame_diffs){
    all_intensity_history.clear(); all_intensity_current.clear(); all_history_frame_diffs.clear();
    ifstream c_file (csv_filename); string value;
    int prev_CFID = 1; int columns = 9; int prev_HKFID = -1;
    vector<vector<float> > CFID_intensity_history, CFID_intensity_current;
    vector<int> CFID_history_frame_diffs;
    vector<float> CFID_HKFID_intensity_history, CFID_HKFID_intensity_current;
    bool done_all_lines = false;
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
                            CFID_HKFID_intensity_history.clear(); CFID_HKFID_intensity_current.clear();
                            all_intensity_history.push_back(CFID_intensity_history);
                            all_intensity_current.push_back(CFID_intensity_current);
                            all_history_frame_diffs.push_back(CFID_history_frame_diffs);
                            CFID_intensity_history.clear(); CFID_intensity_current.clear(); CFID_history_frame_diffs.clear();
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
                            CFID_HKFID_intensity_history.clear(); CFID_HKFID_intensity_current.clear();
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
            }
        }
        if (!done_all_lines) getline ( c_file, value);
        else{
            CFID_intensity_history.push_back(CFID_HKFID_intensity_history);
            CFID_intensity_current.push_back(CFID_HKFID_intensity_current);
            CFID_history_frame_diffs.push_back(prev_HKFID);
            all_intensity_history.push_back(CFID_intensity_history);
            all_intensity_current.push_back(CFID_intensity_current);
            all_history_frame_diffs.push_back(CFID_history_frame_diffs);
        }
    }
}

int main(int argc, char **argv){
    CommandLineParser parser(argc, argv, "{@input_path |0|input path can be a camera id, like 0,1,2 or a video filename}");
    parser.printMessage();
    string input_path = "/media/kimsk/Seagate_Expansion/ECCV_Results/video.mp4"; //parser.get<string>(0);
    string correspondence_path = "/media/kimsk/Seagate_Expansion/ECCV_Results/Correspondences.txt";
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
    Mat prev_frame, frame_3, frame;

    // ===== Read Correspondences
    vector<vector<vector<float> > > all_intensity_history, all_intensity_current;
    vector<vector<int> > all_history_frame_diffs;
    ReadCorrespondences(correspondence_path, all_intensity_history, all_intensity_current, all_history_frame_diffs);

    // ===== Setup for IRPC =================================================
    IRPhotoCalib calib(false);
    vector<int> ptx, pty, pptx, ppty, id, idp; 
    vector<float> o, op;
    while(video_in.isOpened()){
        video_in >> frame_3;
        if(frame_3.empty()) continue;
        cvtColor(frame_3, frame, CV_BGR2GRAY);
        
        if (frame_counter>=1){
            PTAB current_params = calib.ProcessCurrentFrame(all_intensity_history[frame_counter-1], all_intensity_current[frame_counter-1], all_history_frame_diffs[frame_counter-1]);
            Mat corrected_frame = frame * (current_params.a-current_params.b) + current_params.b;
            Mat res;
            hconcat(frame, corrected_frame, res);
            imshow(video_name, res);
            waitKey(1);
        }
        else{

        }
        frame_counter++;
    }
}