#include <iostream>
#include "irPhotoCalib.h"
#include<cstdlib>
#include <time.h>
#include <string>
#include <vector>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <boost/filesystem.hpp>

using namespace std;
using namespace cv;

/* TODOS:
 * Handle termination criteria for the optimizers
 * Find paths and constraint points 
*/

// GLobal Variables
int w, h;
Mat imold;
string dir = "/home/manash/work/research/thermoVIO/JPL-thermal-perception/dso_ws/del/full_seq02/";

void readNonKFS(int offset, int upto_kf, IRPhotoCalib &calib, vector<int> &ptx, vector<int> &pty, vector<int> &pptx, vector<int> &ppty, vector<int> &id, vector<int> &idp,
                vector<float> &o, vector<float> &op, int &n_frames, vector<string> &names)
{
  // Reads NonKeyFrames
  //offset: depends on first imNKF+1
  //upto_kf: depends on the max imnew id
  // Setup these variables before loop
  int bw_count = 0;
  int prev_kf_id = -1;  
  int min_frame_id = 0;
  bool firstKF = true;
  int running_shift = 0;
  n_frames = 0;
  vector<int> shift(upto_kf-offset);
  stringstream str_bldr;
  str_bldr.str(""); str_bldr << 'k' << offset; names.push_back(str_bldr.str());
  for(int i=offset-1; i<=upto_kf-2;)
  {
    stringstream imNonKF_name, corr_name;
    imNonKF_name << dir << "imNKF" << i << "_" << bw_count << ".png";
    corr_name << dir << "corrNKF" << i << "_" << bw_count << ".txt";
    
    //std::cout << i << "_" << bw_count << " NKF\n";
    //check if the file exists, if not increment i
    if ( !boost::filesystem::exists( corr_name.str() ) )
    {
        //std::cout << bw_count << " Skipping\n";
        i++; bw_count=0;
        // just made a jump, i need to set shift for the kf i jumped over;
        shift[i+1-offset] = running_shift;
        str_bldr.str(""); str_bldr << 'k' << i+1; names.push_back(str_bldr.str());
        continue;
    }
    str_bldr.str(""); str_bldr << 'n' << i << "_" << bw_count; names.push_back(str_bldr.str());
    //std::cout << str_bldr.str() << std::endl;
        
    Mat imnew = imread(imNonKF_name.str(), CV_LOAD_IMAGE_GRAYSCALE);
    w = imnew.cols; h = imnew.rows;
    
    string line;
    ifstream f ( corr_name.str().c_str() );
    getline(f, line); // simply skip the timestamp line
    while(getline(f, line)) 
    {
        // Split the parts
        vector<string> parts;
        stringstream ss(line);
        while( ss.good() )
        {
            string substr;
            getline( ss, substr, ',' );
            parts.push_back( substr );
        }
        // Parts: 0: oldKFid, oldptu, oldptv, newptu, newptv
        int t_kfid = (int)atof(parts[0].c_str());
        if(t_kfid < offset) continue;
        stringstream imKF_name;
        //std::cout << t_kfid << " Looking for KF\n";
        imKF_name << dir << "imnew" << t_kfid << ".png";
        
        bool add_pair = false;
        if(prev_kf_id != t_kfid) 
        {
            if ( boost::filesystem::exists( imKF_name.str() ) )
            {
                imold = imread(imKF_name.str(), CV_LOAD_IMAGE_GRAYSCALE);
                //std::cout << "Reading New Image\n";
                //std::cout << "Found KF\n";
                add_pair = true;
            }
        }
        else add_pair=true;
        if(add_pair)
        {
            //Yes KF exists
            //std::cout << "Adding Pair KF\n";
            if( t_kfid >= i-2)
            {
                
                //std::cout << t_kfid << " " << i << " Adding Pair KF\n";
                ptx.push_back((int)atof(parts[1].c_str())); pty.push_back((int)atof(parts[2].c_str()));
                pptx.push_back((int)atof(parts[3].c_str())); ppty.push_back((int)atof(parts[4].c_str()));
                id.push_back(t_kfid-offset+shift[t_kfid-offset]); idp.push_back(i-offset+2+running_shift);
                float ov = calib.getKernelO(imold, (int)atof(parts[1].c_str()), (int)atof(parts[2].c_str()), 16);
                float opv = calib.getKernelO(imnew, (int)atof(parts[3].c_str()), (int)atof(parts[4].c_str()), 16);
                o.push_back(ov); op.push_back(opv);
            }
            prev_kf_id = t_kfid;
        }
    }
    running_shift++;// Whenever I process a new nkf, my running_shift increments
    bw_count++;
  }
  n_frames = running_shift+upto_kf-offset;
}

void readKFS(int offset, int upto_kf, IRPhotoCalib &calib, vector<int> &ptx, vector<int> &pty, vector<int> &pptx, vector<int> &ppty, vector<int> &id, vector<int> &idp,
             vector<float> &o, vector<float> &op)
{
  //offset depends on imold, the id from where it becomes continuous
  // upto_kf depends on max imnew id available.
  // Reads KeyFrames
  // Setup these variables before loop
  int prev_kf_id = -1;  
  int min_frame_id = 0;
  for(int i=offset+1; i<=upto_kf;i++)
  {
    stringstream imnew_name;
    imnew_name << dir << "imnew" << i << ".png";
    //std::cout << i << " loading imnew\n";
    Mat imnew = imread(imnew_name.str(), CV_LOAD_IMAGE_GRAYSCALE);
    w = imnew.cols; h = imnew.rows;
    
    for(int j=offset; j<i; j++)
    {
        stringstream imold_name, corr_name;
        imold_name << dir << "imold" << j << ".png";
        //std::cout << j << " loading imold\n";
        corr_name << dir << "corr" << i << "_" << j << ".txt";
        if ( !boost::filesystem::exists( corr_name.str() ) ) continue;
        Mat imold = imread(imold_name.str(), CV_LOAD_IMAGE_GRAYSCALE);
        string line;
        ifstream f ( corr_name.str().c_str() );
        while(getline(f, line)) 
        {
            // Split the parts
            vector<string> parts;
            stringstream ss(line);
            while( ss.good() )
            {
                string substr;
                getline( ss, substr, ',' );
                parts.push_back( substr );
            }
            // Parts: oldu, oldv, newu, newv
            ptx.push_back((int)atof(parts[0].c_str())); pty.push_back((int)atof(parts[1].c_str()));
            pptx.push_back((int)atof(parts[2].c_str())); ppty.push_back((int)atof(parts[3].c_str()));
            id.push_back(j-offset); idp.push_back(i-offset);
            float ov = calib.getKernelO(imold, (int)atof(parts[0].c_str()), (int)atof(parts[1].c_str()), 16);
            float opv = calib.getKernelO(imnew, (int)atof(parts[2].c_str()), (int)atof(parts[3].c_str()), 16);
            o.push_back(ov); op.push_back(opv);
        }
    }
  }
}

int main(){
  std::srand (time(NULL));
  IRPhotoCalib calib;
  calib.e_photo_error = calib.e_photo_error/10.0;
 
  int div = 16;
  // Always old to new. p is the new one. NKF is always new.
  ofstream results_file;
  stringstream results_file_name;
  results_file_name << dir << "results" << ".txt";
  results_file.open(results_file_name.str().c_str());
  
  int start_id = 5; // starting num for imNKF+1
  int end_id = 79; // max id of imnew
  int window_size = 12;
  int nkf_window_size = 12;
  int min_nkf_window_size = 3;
  int min_window_size = window_size/4;
  for(int i=start_id; i<=end_id;)
  {
      vector<int> kfptx, kfpty, kfpptx, kfppty, kfid, kfidp; vector<float> kfo, kfop; vector<double> kfs, kfn, kfa, kfb;
      // Get correspondences for KFs in the defined window_size
      int running_end_id = (i+window_size)>end_id?end_id:(i+window_size);
      int n_frames = running_end_id-i+1;
      if(n_frames < min_window_size) break;
      std::cout << i << " " << running_end_id << std::endl;      
      
      readKFS(i, running_end_id, calib, kfptx, kfpty, kfpptx, kfppty, kfid, kfidp, kfo, kfop);
      std::cout << kfo.size() << " Total correspondences -------------------------------------------------------------------------\n";      
      //for(int j=0; j<kfop.size(); j++){std::cout << kfop[j] << " ";} std::cout << std::endl;
      
      // Run Full Optimizer for KFs in the defined window_size      
      calib.RunGNAOptimizer(kfptx, kfpty, kfpptx, kfppty, kfo, kfop, kfid, kfidp, w, h, div, n_frames, kfs, kfn, kfa, kfb);
      for(int k=0; k<kfa.size(); k++){std::cout << std::exp(kfa[k]) << " ";} std::cout << std::endl;
      //calib.RunGNAOptimizerAB(kfptx, kfpty, kfpptx, kfppty, kfo, kfop, kfid, kfidp, w, h, div, n_frames, kfs, kfn, kfa, kfb);
      //for(int k=0; k<kfa.size(); k++){std::cout << std::exp(kfa[k]) << " ";} std::cout << std::endl;
      
      //Write to File:
      results_file << "Full Optimization for KFs: ," << i << "," << running_end_id << ", " << kfo.size() << " correspondences\n";
      for(int k=0; k<kfn.size(); k++){ results_file << kfn[k] << ","; } results_file << "\n";
      for(int k=0; k<kfa.size(); k++){ results_file << kfa[k] << ","; } results_file << "\n";
      for(int k=0; k<kfb.size(); k++){ results_file << kfb[k] << ","; } results_file << "\n\n\n";
      
      
      // Now loop through smaller nonkfs
      int nkf_start_id = i; int nkf_end_id = running_end_id;
      for(int j=nkf_start_id; j<=nkf_end_id;)
      {
          vector<int> ptx, pty, pptx, ppty, id, idp; vector<float> o, op; vector<double> s, a, b; vector<double> *n;
          //Get correspondences
          int nkf_running_end_id = (j+nkf_window_size)>nkf_end_id?nkf_end_id:(j+nkf_window_size);
          int nkf_n_frames;
          if(nkf_running_end_id-j+1 < min_nkf_window_size) break;
          vector<string> names;
          readNonKFS(j, nkf_running_end_id, calib, ptx, pty, pptx, ppty, id, idp, o, op, nkf_n_frames, names);
          std::cout << o.size() << " Total nkf correspondences\n";
//           for(int k=0; k<o.size(); k++){ if(std::isnan(o[k])) std::cout << o[k] << "o ";} std::cout << std::endl;
//           for(int k=0; k<op.size(); k++){ if(std::isnan(op[k])) std::cout << op[k] << "op ";} std::cout << std::endl;
//           for(int k=0; k<ptx.size(); k++){ if(ptx[k] > w+1 || ptx[k] < 0 ) std::cout << ptx[k] << "ptx ";} std::cout << std::endl;
//           for(int k=0; k<pty.size(); k++){ if(pty[k] > w+1 || pty[k] < 0 ) std::cout << pty[k] << "pty ";} std::cout << std::endl;
//           for(int k=0; k<pptx.size(); k++){ if(pptx[k] > w+1 || pptx[k] < 0 ) std::cout << pptx[k] << "pptx ";} std::cout << std::endl;
//           for(int k=0; k<ppty.size(); k++){ if(ppty[k] > w+1 || ppty[k] < 0 ) std::cout << ppty[k] << "ppty ";} std::cout << std::endl;
          
          // Run AB Optimizer for the defined nkf_window_size
          //n = *kfn;
          calib.RunGNAOptimizerAB(ptx, pty, pptx, ppty, o, op, id, idp, w, h, div, nkf_n_frames, s, kfn, a, b);
          for(int k=0; k<a.size(); k++){std::cout << std::exp(a[k]) << " ";} std::cout << std::endl;
          
          //Write to File:
          results_file << "Partial Optimization for nKFs," << j << "," << nkf_running_end_id << "," << o.size() << ",correspondences\n";
          for(int k=0; k<a.size(); k++){ results_file << a[k] << ","; } results_file << "\n";
          for(int k=0; k<b.size(); k++){ results_file << b[k] << ","; } results_file << "\n";
          for(int k=0; k<names.size(); k++){ results_file << names[k] << ","; } results_file << "\n\n";

          j = nkf_running_end_id;
      }
      results_file << "\n\n";
      
      
      i = running_end_id;
  }
  results_file.close();
  //readKFS(28, 53, calib, kfptx, kfpty, kfpptx, kfppty, kfid, kfidp, kfo, kfop);
  //readNonKFS(offset, upto_kf, calib, ptx, pty, pptx, ppty, id, idp, o, op);
}
