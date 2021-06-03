#include <iostream>
#include "irPhotoCalib.h"
#include<cstdlib>
#include <time.h>
#include <dirent.h>
#include <string>
#include <vector>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <random>
#include <algorithm>
#include <iterator>

using namespace std;
using namespace cv;
DIR *dpdf;
struct dirent *epdf;


/* TODOS:
 * Handle termination criteria for the optimizers
 * Find paths and constraint points 
*/

struct NKFAB{
  int points;
  int iold_id;
  double a;
  double b;
};

bool getHeatMapColor(float value, float *red, float *green, float *blue)
{
  const int NUM_COLORS = 5;
  // static float color[NUM_COLORS][3] = { {0,0,0}, {0,1,1}, {0,0,1}, {0,1,0}, {1,0,0}, {1,1,0}, {1,1,1} };
  //   // A static array of 7 colors:  (black, cyan, blue, green,  red, yellow, white) using {r,g,b} for each.
  // static float color[NUM_COLORS][3] = { {0,0,0}, {0,0,1}, {0,1,1}, {0,1,0}, {1,1,0}, {1,0,0}, {1,1,1} };
  //   // A static array of 7 colors:       (black,   blue,    cyan,   green,   yellow,   red,    white) using {r,g,b} for each.
  
  static float color[NUM_COLORS][3] = { {0,0,0}, {0,0,1}, {0,1,0},  {1,0,0}, {0,0,0} };
    // A static array of 7 colors:       (black,   blue,   green,    red,    white) using {r,g,b} for each.
  
  int idx1;        // |-- Our desired color will be between these two indexes in "color".
  int idx2;        // |
  float fractBetween = 0;  // Fraction between "idx1" and "idx2" where our value is.
  
  if(value <= 0)      {  idx1 = idx2 = 0;            }    // accounts for an input <=0
  else if(value >= 1)  {  idx1 = idx2 = NUM_COLORS-1; }    // accounts for an input >=0
  else
  {
    value = value * (NUM_COLORS-1);        // Will multiply value by 3.
    idx1  = floor(value);                  // Our desired color will be after this index.
    idx2  = idx1+1;                        // ... and before this index (inclusive).
    fractBetween = value - float(idx1);    // Distance between the two indexes (0-1).
  }
    
  *red   = (color[idx2][0] - color[idx1][0])*fractBetween + color[idx1][0];
  *green = (color[idx2][1] - color[idx1][1])*fractBetween + color[idx1][1];
  *blue  = (color[idx2][2] - color[idx1][2])*fractBetween + color[idx1][2];
}

/* Returns a list of files in a directory (except the ones that begin with a dot) */
void GetFilesInDirectory(std::vector<string> &out, const string &directory)
{
  dpdf = opendir(directory.c_str());
  if (dpdf != NULL){
    while (epdf = readdir(dpdf)){
      string filename = epdf->d_name;
      if (filename.find("corrNKF") != string::npos)
        out.push_back(filename);
    }
  }
  closedir(dpdf);
} // GetFilesInDirectory

// for string delimiter
vector<string> split (string s, string delimiter) {
    size_t pos_start = 0, pos_end, delim_len = delimiter.length();
    string token;
    vector<string> res;

    while ((pos_end = s.find (delimiter, pos_start)) != string::npos) {
        token = s.substr (pos_start, pos_end - pos_start);
        pos_start = pos_end + delim_len;
        res.push_back (token);
    }

    res.push_back (s.substr (pos_start));
    return res;
}

void exportCorrectednColorImages(string dir, int inew_id, int nkf_id, double ai, double bi)
{
  stringstream imnew_name, imnew_corrected_name, imnew_corrected_name_c, imnew_name_c;
  imnew_name << dir << "imNKF" << inew_id << "_" << nkf_id << ".png";                           // Input Image
  imnew_corrected_name << dir << "/results/" << "imNKFC" << inew_id << "_" << nkf_id << ".png";                // Corrected GrayScale Image
  imnew_corrected_name_c << dir << "/results/" << "imNKFCi" << inew_id << "_" << nkf_id << ".png";             // Corrected Colormap Image
  imnew_name_c << dir << "/results/" << "imNKFi" << inew_id << "_" << nkf_id << ".png";                       // Input image as Colormap Image
  Mat imnew = imread(imnew_name.str(), CV_LOAD_IMAGE_GRAYSCALE);
  Mat imnewCHM(imnew.rows, imnew.cols, CV_8UC3, Scalar(0,0,0));
  Mat imnewHM(imnew.rows, imnew.cols, CV_8UC3, Scalar(0,0,0));
  Mat imnewC(imnew.rows, imnew.cols, CV_8UC3, Scalar(0,0,0));

  vector<int> compression_params;
  compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
  compression_params.push_back(0);

  for(int r=0; r<imnew.rows; r++)
  {
    for(int c=0; c<imnew.cols; c++)
    {
      float o = imnew.at<uchar>(r,c)/255.0;
      float co = o*(ai-bi)+bi;
      co = fmod(co, 1.0);
      float hr,hg,hb;
      getHeatMapColor(co,&hr,&hg,&hb);
      
      imnewCHM.at<Vec3b>(r,c)[2] = (u_int8_t)(hr*255);
      imnewCHM.at<Vec3b>(r,c)[1] = (u_int8_t)(hg*255);
      imnewCHM.at<Vec3b>(r,c)[0] = (u_int8_t)(hb*255);

      imnewC.at<Vec3b>(r,c)[2] = (u_int8_t)(co*255);
      imnewC.at<Vec3b>(r,c)[1] = (u_int8_t)(co*255);
      imnewC.at<Vec3b>(r,c)[0] = (u_int8_t)(co*255);

      getHeatMapColor(o,&hr,&hg,&hb);
      imnewHM.at<Vec3b>(r,c)[2] = (u_int8_t)(hr*255);
      imnewHM.at<Vec3b>(r,c)[1] = (u_int8_t)(hg*255);
      imnewHM.at<Vec3b>(r,c)[0] = (u_int8_t)(hb*255);
    }
  }
  imwrite(imnew_corrected_name.str(), imnewC, compression_params);
  imwrite(imnew_corrected_name_c.str(), imnewCHM);
  imwrite(imnew_name_c.str(), imnewHM);
}

void exportGains(vector<double> & all_aips, vector<double> & all_bips, vector<vector<NKFAB> > & combinedGains, int startp1, string dir, string PT_location){
  ofstream result_file;
  result_file.open(PT_location);
  // ==============================     Correcting the image   ===================================================
  for(int i=0; i<combinedGains.size(); i++)
  {
    int inew_id = startp1+i;
    // Export the KF
    //exportCorrectednColorImages(dir, inew_id, 0, all_aips[i], all_bips[i]);
    result_file << inew_id << ',' << 0 << ',' << all_aips[i] << ',' << all_bips[i] << '\n';
    for(int j=1; j<combinedGains[i].size(); j++) // For each NKF
    {
      // Export the NKF
      //exportCorrectednColorImages(dir, inew_id, j, combinedGains[i][j].a, combinedGains[i][j].b);
      result_file << inew_id << ',' << j << ',' << combinedGains[i][j].a << ',' << combinedGains[i][j].b << '\n';
    }
  }
  result_file.close();
}

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

vector<vector<NKFAB> > combineGains(vector<double> & all_aips, vector<double> & all_bips, vector<vector<vector<NKFAB> > > & abNKFs, int startp1){
  vector<vector<NKFAB> > combinedNKFGains(abNKFs.size());
  for(int i=0; i<abNKFs.size(); i++)
  {
    vector<NKFAB> thisBase;
    thisBase.push_back({0,0,0,0}); // Dummy for j = 0
    double ai = all_aips[i]; double bi = all_bips[i]; // Params to this frame relative to the first frame
    double prevwa = 1; double prevwb = 0;
    for(int j=1; j<abNKFs[i].size(); j++) // For each NKF
    {
      double wa = 0; double wb = 0; int tpoints = 0;
      for(int k=0; k<abNKFs[i][j].size(); k++) // For each iold
      {
        if (abNKFs[i][j][k].points==0) continue;
        if (abNKFs[i][j][k].iold_id < startp1) continue;
        
        if(abNKFs[i][j][k].iold_id == i+startp1)
        {
          wa += abNKFs[i][j][k].a*abNKFs[i][j][k].points;
          wb += abNKFs[i][j][k].b*abNKFs[i][j][k].points;
          tpoints += abNKFs[i][j][k].points;
          //cout << "Base Case: " <<  abNKFs[i][j][k].points << " : " << abNKFs[i][j][k].a << " " << abNKFs[i][j][k].b << endl;
        }
        else{
          // Get Gain of i with respect to iold
          double aiold = all_aips[abNKFs[i][j][k].iold_id-startp1];
          double biold = all_bips[abNKFs[i][j][k].iold_id-startp1];
          double aioldi, bioldi;
          getRelativeGains(aiold, biold, ai, bi, aioldi, bioldi);

          // Get Estimated Gain of NKF w.r.t iold
          double aioldn = abNKFs[i][j][k].a;
          double bioldn = abNKFs[i][j][k].b;
          double ain, bin;
          getRelativeGains(aioldi, bioldi, aioldn, bioldn, ain, bin);

          wa += ain*abNKFs[i][j][k].points;
          wb += bin*abNKFs[i][j][k].points;
          tpoints += abNKFs[i][j][k].points;
          //cout << ain << " " << bin << " : " << abNKFs[i][j][k].iold_id << " " << abNKFs[i][j][k].points << endl;
        }
      }
      wa = wa / tpoints; wb = wb / tpoints;

      // if (wa - wb < 1.0){
      //   double push = (1.0-(wa-wb))*0.025;
      //   wa=wa+push; wb=wb-push;
      // }
      // wa= wa - (wa-1.0)*0.075;
      // wb= wb - (wb-0.0)*0.075;

      // if (prevwa!=wa){
      //   double diff = 0.0002/(prevwa-wa);
      //   double awa = wa;
      //   if(diff<0) awa = std::max(prevwa,wa+diff);
      //   else awa = std::min(prevwa,wa+diff);
      //   if (awa != prevwa)  cout << wa << " " << awa << endl;
      //   // cout << diff <<"," << prevwa-wa << endl;
      //   wa = awa; 
      // }

      // if (prevwb!=wb){
      //   double diff = 0.0002/(prevwb-wb);
      //   double awb = wb;
      //   if(diff<0) awb = std::max(prevwb,wb+diff);
      //   else awb = std::min(prevwb,wb+diff);
      //   if (awb != prevwb)  cout << wb << " " << awb << endl;
      //   // cout << diff <<"," << prevwb-wb << endl;
      //   wb = awb; 
      // }

      // wa = wa - (wa-prevwa)*0.65;
      // wb = wb - (wb-prevwb)*0.65;
      // wa= wa - (wa-1.0)*0.075;
      // wb= wb - (wb-0.0)*0.075;
      

      // Transform it to global
      double owa, owb;
      chainGains(ai, bi, wa, wb, owa, owb);
      thisBase.push_back({tpoints, 0, owa, owb});
      prevwa = wa;
      prevwb = wb;
    }
    combinedNKFGains[i] = thisBase;
  }
  return combinedNKFGains;
}

vector<vector<vector<NKFAB> > > ProcessNonKeyFramesSupport(int startp1, int end, string dir, int max_history){
  std::srand (time(NULL));
  std::random_device rd;
  std::mt19937 g(rd());
  IRPhotoCalib calib;
  calib.e_photo_error = calib.e_photo_error/10.0;
  vector<int> ptx, pty, pptx, ppty, id, idp; 
  vector<float> o, op;
  int w, h;
  int div = 8;
  int max_points_per_frame = 600;
  int min_points_per_frame = 5;
  vector<vector<int> > nkf_frames(end-(startp1)+1); // Tis hold the nkfs for each base_kf. Note corrNKF<base>_<nkf>_<iold>. This should also include startp1 as base 
  
  // Always old to new. p is the new one

  int n_frames = 0;
  vector<string> file_names;
  vector<pair<int, int> > corrNKF_fleIDS;
  vector<vector<vector<int> > > iolds_for_NKFS( end-(startp1)+1 ); // Initialize with the number of keyframes that serve as bases for NKFs starting from 1 nkf of startp1-1
  GetFilesInDirectory(file_names, dir);
  // std::sort(file_names.begin(), file_names.end());
  for (int i=0; i<file_names.size();i++){
    vector<string> v = split (file_names[i].substr(7), "_");
    int base_KF = stoi(v[0]);
    if (base_KF <= end && base_KF >= startp1)
      nkf_frames[stoi(v[0])-startp1].push_back(stoi(v[1]));
  }
  for (int i=0; i<nkf_frames.size(); i++){
    sort( nkf_frames[i].begin(), nkf_frames[i].end() );
    nkf_frames[i].erase( unique( nkf_frames[i].begin(), nkf_frames[i].end() ), nkf_frames[i].end() );
    for (int j=0; j<nkf_frames[i].size(); j++){
      cout << "base " << startp1+i << " " << nkf_frames[i][j] << endl;
      corrNKF_fleIDS.push_back(make_pair(startp1+i, nkf_frames[i][j]));
      n_frames++;
    }
    iolds_for_NKFS[i] = vector<vector<int> > (nkf_frames[i].size());
  }
  for (int i=0; i<file_names.size();i++){
    vector<string> v = split (file_names[i].substr(7), "_");
    int ioldKF = stoi(v[2].substr(0,v[2].size()-4));
    int base_KF = stoi(v[0]);
    if (ioldKF >= startp1 && base_KF <= end && base_KF >= startp1) /// ASSUMPTION: We don't have history older than startp1-1
      iolds_for_NKFS[base_KF-startp1][stoi(v[1])].push_back( ioldKF );
  }
  int numKFbases = iolds_for_NKFS.size();
  //cout << numKFbases << "numkfbases\n";
  // Sort imolds in decreasing order
  for (int i=0; i<numKFbases; i++)
  {
    for (int j=0; j<iolds_for_NKFS[i].size(); j++)
    {
      sort( iolds_for_NKFS[i][j].rbegin(), iolds_for_NKFS[i][j].rend() );       // In reverse order
      // for(int k=0; k<vec.size()-max_history; k++)
      //   vec.pop_back();
    }
  }

  /// Now we arrange all the KFS first then then the nonKFS. id and idp are just used to identify the correspondences. So we can use any convention
  vector<vector<vector<int> > > running_ids;
  running_ids = iolds_for_NKFS;
  int running_id = numKFbases;
  for (int i=0; i<numKFbases; i++) // For each base frame
  {
    for (int j=1; j<iolds_for_NKFS[i].size(); j++) // For each nonKFs except for 0 which is the KF itself
    {
      int inew_id = i+startp1;
      stringstream imnew_name;
      imnew_name << dir << "imNKF" << inew_id << "_" << j << ".png";
      Mat imnew = imread(imnew_name.str(), CV_LOAD_IMAGE_GRAYSCALE); // the nKF image
      for (int k=0; k < iolds_for_NKFS[i][j].size(); k++)
      {
        if (k>=max_history) break;
        int iold_id = iolds_for_NKFS[i][j][k];
        stringstream imold_name, corr_name;
        
        imold_name << dir << "imold" << iold_id << ".png";
        corr_name << dir << "corrNKF" << inew_id << "_" << j << "_" << iold_id << ".txt";
        cout << corr_name.str() << " Looking for this file\n";
        Mat imold = imread(imold_name.str(), CV_LOAD_IMAGE_GRAYSCALE); //the base KF image
        w = imnew.cols; h = imnew.rows;

        string line;
        ifstream f ( corr_name.str().c_str() );
        while(getline(f, line)) {
          // Split the parts
          vector<string> parts;
          stringstream ss(line);
          while( ss.good() )
          {
              string substr;
              getline( ss, substr, ',' );
              parts.push_back( substr );
          }
          //cout << parts[4] << " " << inew_id << "," << j << "," << iold_id << " | " << running_id << "," << iold_id-startp1<< endl;
          ptx.push_back((int)atof(parts[0].c_str())); pty.push_back((int)atof(parts[1].c_str()));
          pptx.push_back((int)atof(parts[2].c_str())); ppty.push_back((int)atof(parts[3].c_str()));
          id.push_back(iold_id-startp1); idp.push_back(running_id);
          float ov = calib.getKernelO(imold, (int)atof(parts[0].c_str()), (int)atof(parts[1].c_str()), 32);
          float opv = calib.getKernelO(imnew, (int)atof(parts[2].c_str()), (int)atof(parts[3].c_str()), 32);
          o.push_back(ov); op.push_back(opv);
        }
        running_ids[i][j][k] = running_id;
        running_id++;
      }
    }
  }

  vector<vector<vector<NKFAB> > > absNKF (end-startp1+1); // each NKF frame is stored in a vector of its bases. And each contain a vector of num points and abs
  // Perform Pairwise Calibration for each NKF and its bases
  for (int i=0; i<numKFbases; i++) // For each base frame
  {
    vector<vector<NKFAB> > thisNKFdata;
    vector<NKFAB> dummy; dummy.push_back({0,0,0,0});
    thisNKFdata.push_back(dummy); // Push a dummy to take the position of 0 
    for (int j=1; j<iolds_for_NKFS[i].size(); j++) // For each nonKFs except for 0 which is the KF itself
    {
      vector<NKFAB> thisiold;
      for (int k=0; k<iolds_for_NKFS[i][j].size(); k++)
      {
        if (k>=max_history) break;
        int inew_id = i+startp1; int iold_id = iolds_for_NKFS[i][j][k];
        double aip, bip; vector<double> mean_devs;
        int points = calib.RansacGains(ptx,pty,pptx,ppty,o,op,id,idp,w,h,div,iold_id-startp1,running_ids[i][j][k],startp1,dir,aip,bip,mean_devs);
        if (points!=0){
          NKFAB d = {points, iold_id, aip, bip};
          thisiold.push_back( d );
        }
      }
      thisNKFdata.push_back(thisiold);
    }
    absNKF[i]=thisNKFdata;
  }
  return absNKF;
}

int main(int argc, char *argv[]){
  // Dataset Specific Params
  int end = atoi(argv[5]); // TODO: Change This ==================================================================================
  int offset = atoi(argv[4]); // min
  string dataset_name = argv[1];
  // Run Related Params
  bool estimate_PT = false;
  bool estimate_PS = false;
  bool no_drift = false;
  cout << offset << " " << end << endl;
  if (strcmp(argv[2], "both") == 0){estimate_PS=true; estimate_PT=true;}
  else if (strcmp(argv[2], "PT") == 0){estimate_PT=true;}
  else if (strcmp(argv[2], "PS") == 0){estimate_PS=true;}

  if (strcmp(argv[3], "true") == 0) no_drift=true;

  // Final Derived Dataset Related Params
  string eccv_dir = "/mnt/0C7A58E87A58CFD6/ECCV_Results/"; // leave end /
  string version_name = dataset_name+"_1";

  string dir = eccv_dir+"Initial_DSO_Tracks/"+version_name+"/";
  if(no_drift) version_name+="_nodrift";
  string PTKF_location = eccv_dir+dataset_name+"/PTKF_"+version_name+".txt";
  string PT_correspondence = eccv_dir+dataset_name+"/CorresPT_"+version_name+".txt";
  string PS_corresopondence = eccv_dir+dataset_name+"/CorresPS_"+version_name+".txt";
  string PS_location = eccv_dir+dataset_name+"/PS_"+version_name+".txt";
  string PT_location = eccv_dir+dataset_name+"/PT_"+version_name+".txt";
  string A_location = eccv_dir+dataset_name+"/A_"+version_name+".txt";

  std::srand (time(NULL));
  std::random_device rd;
  std::mt19937 g(rd());
  IRPhotoCalib calib;
  calib.e_photo_error = calib.e_photo_error/10.0;
  vector<int> ptx, pty, pptx, ppty, id, idp; 
  vector<float> o, op;
  int w, h;
  int n_frames = end-offset;
  int div = 8;
  int max_points_per_frame = 600; //600 in general, but lesser 20 or 30 for spatial
  int min_points_per_frame = 4;
  int max_history = 5;
  vector<int> num_points(n_frames+1, 0);
  vector<int> best_iolds(n_frames+1,0);

  vector<vector<vector<NKFAB> > > absNKF;
  if(estimate_PT) absNKF = ProcessNonKeyFramesSupport(offset, end, dir, max_history);
  
  // Always old to new. p is the new one
  // Collect correspondence points and images
  for(int inew=1; inew<n_frames+1; inew++){
    int points_this_frame = 0;
    int best_iold = 1; int max_points=0;
    for (int iold=1; iold<=min(max_history,inew); iold++ ){
      int inew_id = inew+offset; int iold_id = inew_id-iold;
      stringstream imnew_name, imold_name, corr_name;
      imnew_name << dir << "imnew" << inew_id << ".png";
      imold_name << dir << "imold" << iold_id << ".png";
      corr_name << dir << "corr" << inew_id << "_" << iold_id << ".txt";
      cout << corr_name.str() << " Looking for this file\n";
      Mat imnew = imread(imnew_name.str(), CV_LOAD_IMAGE_GRAYSCALE);
      Mat imold = imread(imold_name.str(), CV_LOAD_IMAGE_GRAYSCALE);
      w = imnew.cols; h = imnew.rows;
      
      vector<KeyPoint> kpnews, kpolds;
      string line;
      ifstream f ( corr_name.str().c_str() );
      vector<int> this_ptx, this_pty, this_pptx, this_ppty, this_id, this_idp, pickid; 
      vector<float> this_o, this_op;
      int d_counter = 0;
      while(getline(f, line)) {
        // Split the parts
        vector<string> parts;
        stringstream ss(line);
        while( ss.good() )
        {
            string substr;
            getline( ss, substr, ',' );
            parts.push_back( substr );
        }
        this_ptx.push_back((int)atof(parts[0].c_str())); this_pty.push_back((int)atof(parts[1].c_str()));
        this_pptx.push_back((int)atof(parts[2].c_str())); this_ppty.push_back((int)atof(parts[3].c_str()));
        this_id.push_back(inew-iold); this_idp.push_back(inew);
        // o.push_back( imold.at<uchar>((int)atof(parts[1].c_str()), (int)atof(parts[0].c_str()))/255.0 );
        // op.push_back( imnew.at<uchar>((int)atof(parts[3].c_str()), (int)atof(parts[2].c_str()))/255.0 );
        float ov = calib.getKernelO(imold, (int)atof(parts[0].c_str()), (int)atof(parts[1].c_str()), 32);
        float opv = calib.getKernelO(imnew, (int)atof(parts[2].c_str()), (int)atof(parts[3].c_str()), 32);
        this_o.push_back(ov); this_op.push_back(opv);
        pickid.push_back(d_counter);
        d_counter++;
      }
      if(max_points < this_ptx.size()){
        max_points = this_ptx.size();
        num_points[inew]=this_ptx.size();
        best_iold = iold;
      }
      std::shuffle(pickid.begin(), pickid.end(), g);
      int added_from_this_old_frame = 0;
      for (int j=0; j<max_points_per_frame-points_this_frame; j++)
      {
        if(j >= this_ptx.size()) { 
          //std::cout << "Not sufficient points for scene " << inew_id << " " << iold_id << " " << this_ptx.size() << "\n"; 
          break; }
        ptx.push_back( this_ptx[pickid[j]] ); pty.push_back( this_pty[pickid[j]] );
        pptx.push_back( this_pptx[pickid[j]] ); ppty.push_back( this_ppty[pickid[j]] );
        id.push_back( this_id[pickid[j]] ); idp.push_back( this_idp[pickid[j]] );
        o.push_back( this_o[pickid[j]] ); op.push_back( this_op[pickid[j]] );
        added_from_this_old_frame++;
      }
      points_this_frame += added_from_this_old_frame;
      if (points_this_frame == max_points_per_frame){
        best_iolds[inew] = best_iold;
        break;
      }
      // if (points_this_frame > min_points_per_frame){
      //   best_iolds[inew] = best_iold;
      //   break;
      // }
    }
    if(points_this_frame < min_points_per_frame){std::cout << inew+offset << "A frame with less points in total " << points_this_frame << "\n"; }
    best_iolds[inew] = best_iold;
  }

  calib.exportCorrespondences(ptx, pty, pptx, ppty, o, op, id, idp, PT_correspondence);
  
  vector<double> s, n, a, b;
  vector<double> ind_aips, ind_bips;
  vector<vector<double> > ind_mean_devs;
  for(int i=1; i<n_frames+1;i++)
  {
    double aip, bip; vector<double> mean_devs;
    calib.RansacGains(ptx,pty,pptx,ppty,o,op,id,idp,w,h,div,i-best_iolds[i],i,offset,dir,aip,bip,mean_devs);
    ind_aips.push_back(aip); ind_bips.push_back(bip);
    ind_mean_devs.push_back(mean_devs);
    //std::cout << mean_devs[0] << " " << mean_devs[1] << " " << mean_devs[2] << " " << mean_devs[3] << std::endl;
  }
  vector<double> all_aips, all_bips;
  all_aips.push_back(1.0); all_bips.push_back(0.0); //// ############## all_aips = [1] + chained(ind_aips)

  double t_la_k; double t_lb_k = 0.0; int kid = 0;
  double t_ua_g = 1.0; double t_ub_g; int gid = 0;
  for(int k=1; k<n_frames+1; ++k)
  {
    double ko_b_kn = ind_bips[k-1];
    //if (std::fabs(k_b_k1-0.0) < 0.05 ) {k_b_k1=0.0; ind_bips[k]=0.0;}
    //if (k_b_k1 < 0.05 && k_b_k1 > 0) {k_b_k1=0.0; ind_bips[k]=0.0;}
    double ko_a_kn = ind_aips[k-1];
    //if (std::fabs(k_a_k1-1.0) < 0.05 ) {k_a_k1=1.0; ind_aips[k]=1.0;}
    //if (k_a_k1 < 1.0 && k_a_k1 > 1.0-0.05 ) {k_a_k1=1.0; ind_aips[k]=1.0;}
    int ko = k-best_iolds[k];
    double t_a_ko = all_aips[ko]; double t_b_ko = all_bips[ko];

    double t_b_kn = t_b_ko + (t_a_ko - t_b_ko) * ko_b_kn;
    double t_a_kn = ((t_a_ko-t_b_ko)*(ko_a_kn-ko_b_kn))+t_b_kn;

    //if (t_a_kn - t_b_kn < 1.0){
      double push = (1.0-(t_a_kn-t_b_kn))*0.025;
      t_a_kn=t_a_kn+push; t_b_kn=t_b_kn-push;
    //}
    if (!no_drift){
      t_a_kn= t_a_kn - (t_a_kn-1.0)*0.025;
      t_b_kn= t_b_kn - (t_b_kn-0.0)*0.025;
    }

    // double push = 0;
    // if (t_a_kn - t_b_kn < 1.0){
    //   push = (1.0-(t_a_kn-t_b_kn))*0.025;
    // }
    // if (no_drift){
    //   t_a_kn= t_a_kn + push;
    //   t_b_kn= t_b_kn - push;
    // }
    // else{
    //   t_a_kn= t_a_kn - (t_a_kn-1.0)*0.025 + push;
    //   t_b_kn= t_b_kn - (t_b_kn-0.0)*0.025 - push;
    // }

    all_aips.push_back(t_a_kn); all_bips.push_back(t_b_kn);

    if (t_b_kn < t_lb_k){t_lb_k=t_b_kn; t_la_k=t_a_kn; kid=k;}
    if (t_a_kn > t_ua_g){t_ua_g=t_a_kn; t_ub_g=t_b_kn; gid=k;}
  }
  double cb = 0; double ca = 0;
  if (kid != 0){
    cb = (-1.0)/((t_ua_g/t_lb_k)-1.0);
    ca = (-cb*t_lb_k+cb+t_lb_k)/t_lb_k;
  }

  vector<double> s_all_aips, s_all_bips;
  for(int k=0; k<all_aips.size(); ++k)
  {
    double t_b_k = all_bips[k];
    double t_a_k = all_aips[k];

    double o_b_k = t_b_k*(1.0-ca-cb)+cb;
    double o_a_k = (t_a_k-t_b_k)*(1.0-ca-cb) + o_b_k;

    s_all_aips.push_back(o_a_k); s_all_bips.push_back(o_b_k);
    a.push_back(o_a_k); b.push_back(o_b_k);
  }
  calib.exportKFPT(all_aips, all_bips, PTKF_location);

  // std::cout << ca << " ca " << cb << kid << " kid " << gid <<"\n\n";
  // std::copy(all_aips.begin(), all_aips.end(), std::ostream_iterator<double>(std::cout, ", "));
  // std::cout << "aips\n";
  // std::copy(all_bips.begin(), all_bips.end(), std::ostream_iterator<double>(std::cout, ", "));
  // std::cout <<"bips\n\n";
  // std::copy(ind_aips.begin(), ind_aips.end(), std::ostream_iterator<double>(std::cout, ", ")); 
  // std::cout << "inda\n";
  // std::copy(ind_bips.begin(), ind_bips.end(), std::ostream_iterator<double>(std::cout, ", "));  std::cout<< std::endl;
  // std::cout <<"bips\n\n";
  // std::copy(s_all_aips.begin(), s_all_aips.end(), std::ostream_iterator<double>(std::cout, ", "));
  // std::cout << "saips\n";
  // std::copy(s_all_bips.begin(), s_all_bips.end(), std::ostream_iterator<double>(std::cout, ", "));
  // std::cout <<"sbips\n\n";
  // std::copy(num_points.begin(), num_points.end(), std::ostream_iterator<int>(std::cout, ", "));
  // std::cout <<"numpoints\n\n";
  // std::copy(best_iolds.begin(), best_iolds.end(), std::ostream_iterator<int>(std::cout, ", "));
  // std::cout <<"iolds\n\n";

  if(estimate_PS) calib.EstimateSpatialParameters(ptx, pty, pptx, ppty, o, op, id, idp, all_aips, all_bips, w, h, div, 
                                                 PS_corresopondence, PS_location, A_location);
  vector<vector<NKFAB> > combinedGains;
  if(estimate_PT) combinedGains = combineGains(all_aips, all_bips, absNKF, offset);
  if(estimate_PT) exportGains(all_aips, all_bips, combinedGains, offset, dir, PT_location);
  return 0;

  // Correcting the image
  double max_co = 0.0; double min_co = 1.0;
  for(int i=1; i<n_frames+1; i++)
  {
    stringstream imnew_name, imnew_corrected_name, imnew_corrected_name_c, imnew_name_c;
    imnew_name << dir << "imnew" << i+offset << ".png";
    imnew_corrected_name << dir << "imnewC" << i+offset << ".png";
    imnew_corrected_name_c << dir << "imnewCi" << i+offset << ".png";
    imnew_name_c << dir << "imnewi" << i+offset << ".png";
    Mat imnew = imread(imnew_name.str(), CV_LOAD_IMAGE_GRAYSCALE);
    Mat imnewCHM(imnew.rows, imnew.cols, CV_8UC3, Scalar(0,0,0));
    Mat imnewHM(imnew.rows, imnew.cols, CV_8UC3, Scalar(0,0,0));
    Mat imnewC(imnew.rows, imnew.cols, CV_8UC3, Scalar(0,0,0));
    double ai=all_aips[i]; double bi=all_bips[i];
    for(int r=0; r<imnew.rows; r++)
    {
      for(int c=0; c<imnew.cols; c++)
      {
        float o = imnew.at<uchar>(r,c)/255.0;
        float co = o*(ai-bi)+bi;
        //co = (co-0.25)/(0.75-0.25);
        co = fmod(co, 1.0);
        //co = co/0.955586;
        //if(co>1.0) {co = 1.0-co;}
        //std::cout<<"large\n";}
        //else if(co<0.0) {co = 0.0;}
        //std::cout<<"small\n";}
        if (max_co < co) max_co = co;
        if (min_co > co) min_co = co;
        float hr,hg,hb;
        getHeatMapColor(co,&hr,&hg,&hb);
        
        imnewCHM.at<Vec3b>(r,c)[2] = (u_int8_t)(hr*255);
        imnewCHM.at<Vec3b>(r,c)[1] = (u_int8_t)(hg*255);
        imnewCHM.at<Vec3b>(r,c)[0] = (u_int8_t)(hb*255);

        imnewC.at<Vec3b>(r,c)[2] = (u_int8_t)(co*255);
        imnewC.at<Vec3b>(r,c)[1] = (u_int8_t)(co*255);
        imnewC.at<Vec3b>(r,c)[0] = (u_int8_t)(co*255);

        getHeatMapColor(o,&hr,&hg,&hb);
        imnewHM.at<Vec3b>(r,c)[2] = (u_int8_t)(hr*255);
        imnewHM.at<Vec3b>(r,c)[1] = (u_int8_t)(hg*255);
        imnewHM.at<Vec3b>(r,c)[0] = (u_int8_t)(hb*255);
      }
    }
    cout << "writing to image\n";
    imwrite(imnew_corrected_name.str(), imnewC);
    // Mat coimnew_c,imnew_c;
    // applyColorMap(coimnew, coimnew_c, COLORMAP_JET);
    // applyColorMap(imnew, imnew_c, COLORMAP_JET);
    imwrite(imnew_corrected_name_c.str(), imnewCHM);
    // imwrite(imnew_name_c.str(), imnew_c);
    imwrite(imnew_name_c.str(), imnewHM);
  }
  float hr,hg,hb;
  getHeatMapColor(0.0,&hr,&hg,&hb);
  std::cout << hr << " " << hg << " " << hb << std::endl;
  getHeatMapColor(1.0,&hr,&hg,&hb);
  std::cout << hr << " " << hg << " " << hb << std::endl;
  std::cout << max_co << " co " << min_co << std::endl;

  
  return 0;
  calib.RunOptimizerCMU(ptx, pty, pptx, ppty, o, op, id, idp, w, h, div, n_frames, s, n, a, b);
  std::copy(a.begin(), a.end(), std::ostream_iterator<double>(std::cout, ", "));
  std::cout << "\nFull A\n";
  std::copy(b.begin(), b.end(), std::ostream_iterator<double>(std::cout, ", "));
  std::cout <<"\nFull B\n\n";
  //calib.RunGainOptimizer(ptx, pty, pptx, ppty, o, op, id, idp, w, h, div, n_frames, n, a, b);
  
  // // Printing frame exposures
  // std::cout << "Up and Low limits for each frames\n";
  // for (int i=0; i<n_frames; i++)
  // {
  //   std::cout << "AB," << i+offset << "," << std::exp(a[i]) << "," << a[i]  << "," << b[i] << std::endl;
  // }
  
  //return 0;
  
  // Displaying the pixel response
  Mat imN((h/div)+1, (w/div)+1, CV_8UC1, Scalar(0));
  Mat imMask((h/div)+1, (w/div)+1, CV_8UC1, Scalar(0));
  double maxn = -99; double minn = 99;
  for (int i=0; i<n.size(); i++)
  {
    if(!(n[i] == 0.000001))
    {
      if( n[i] < minn ) minn = n[i];
      if(n[i] > maxn) maxn = n[i];
    }
  }
  //cout << maxn << " " << minn << "max min\n";
  for (int i=0; i<n.size(); i = i+1)
  {
    if(!(n[i] == 0.000001))
    {
      //double sxy = exp(s[i])+n[i] - (maxs - 1.0);
      double nxy = (n[i]-minn)/(maxn-minn);
      int x = i%imN.cols;
      int y = i/imN.cols;
      imN.at<uchar>(y,x) = (int)(nxy*255.0);
    }
    else
    {
      int x = i%imN.cols;
      int y = i/imN.cols;
      imMask.at<uchar>(y,x) = 255;
    }
  }
  Mat imNi, imNis;
  inpaint(imN, imMask, imNi, 6, CV_INPAINT_NS);
  GaussianBlur( imNi, imNis, Size( 3, 3 ), 0, 0 );
  imwrite("N.png", imN); imwrite("Ni.png", imNi); imwrite("Nis.png", imNis);

  // // Printing corrected O values
  // for(int i=0; i<ptx.size(); i++)
  // {
  //   cout << o[i] << " " << op[i] << " _ " << calib.getCorrected(o[i], ptx[i], pty[i], a, b, s, n, id[i], w, h, div, imNis, minn, maxn) << " " << calib.getCorrected(op[i], pptx[i], ppty[i], a, b, s, n, idp[i], w, h, div, imNis, minn, maxn) << endl;
  // }

  // Correcting the image
  for(int i=1; i<n_frames; i++)
  {
    stringstream imnew_name, imnew_corrected_name, imnew_corrected_name_c, imnew_name_c;
    imnew_name << dir << "imnew" << i+offset << ".png";
    imnew_corrected_name << dir << "imnewC" << i+offset << ".png";
    imnew_corrected_name_c << dir << "imnewCi" << i+offset << ".png";
    imnew_name_c << dir << "imnewi" << i+offset << ".png";
    Mat imnew = imread(imnew_name.str(), CV_LOAD_IMAGE_GRAYSCALE);
    Mat coimnew; imnew.copyTo(coimnew);
    for(int r=0; r<imnew.rows; r++)
    {
      for(int c=0; c<imnew.cols; c++)
      {
        float o = imnew.at<uchar>(r,c)/255.0;
        float co = calib.getCorrected(o, c, r, a, b, s, n, i, w, h, div, imNis, minn, maxn);
        if(co>1.0) co = 1.0;
        else if(co<0.0) co = 0.0;
        coimnew.at<uchar>(r,c) = (int)(co*255.0);
      }
    }
    imwrite(imnew_corrected_name.str(), coimnew);
    Mat coimnew_c,imnew_c;
    applyColorMap(coimnew, coimnew_c, COLORMAP_JET);
    applyColorMap(imnew, imnew_c, COLORMAP_JET);
    imwrite(imnew_corrected_name_c.str(), coimnew_c);
    imwrite(imnew_name_c.str(), imnew_c);
  }
}