#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

vector<Point2f> Points(vector<KeyPoint> keypoints);
bool isInsideFrame(int x, int y, int w, int h);
float getKernelO(cv::Mat& im, int x, int y, int k);
void GetFrameIntensities(std::vector<cv::KeyPoint>& kp, cv::Mat & image, int window_size, std::vector<float>& out_intensities);
float GetInliers(Mat& prev_desc, vector<KeyPoint>& prev_kp,
                Mat& desc, vector<KeyPoint>& kp,
                Ptr<DescriptorMatcher> matcher,
                vector<KeyPoint>& out_inliers_prev,
                vector<KeyPoint>& out_inliers);

vector<Point2f> Points(vector<int>& ids, vector<KeyPoint>& keypoints)
{
    vector<Point2f> res;
    for(unsigned i = 0; i < ids.size(); i++) {
        res.push_back(keypoints[ids[i]].pt);
    }
    return res;
}

bool isInsideFrame(int x, int y, int w, int h)
{
  if(x>=0 && x<w && y>=0 && y<h)
    return true;
  return false;
}

float getKernelO(cv::Mat& im, int x, int y, int k)
{
  float o = 0;
  int counter = 0;
  for(int r=-k/2.0; r<k/2.0; r++)
  {
    for(int c=-k/2.0; c<k/2.0; c++)
    {
      if(isInsideFrame(x+c, y+r, im.cols, im.rows))
      {
        o += (float)(im.at<uchar>(y+r, x+c))/255.0;
        counter++;
      }
    }
  }
  return o/counter;
}

void GetFrameIntensities(std::vector<cv::KeyPoint>& kp, cv::Mat & image, int window_size, std::vector<float>& out_intensities)
{
    out_intensities.clear();
    for(size_t i=0; i<kp.size(); i++){
        float o = getKernelO(image,kp[i].pt.x, kp[i].pt.y, window_size);
        out_intensities.push_back(o);
    }
}

float GetInliers(Mat& prev_desc, vector<KeyPoint>& prev_kp,
                Mat& desc, vector<KeyPoint>& kp,
                Ptr<DescriptorMatcher> matcher,
                double ransac_thresh, double nn_match_ratio,
                vector<int>& out_inliers_prev,
                vector<int>& out_inliers){
    vector<vector<DMatch> > matches;
    vector<int> matched1, matched2;
    matcher->knnMatch(prev_desc, desc, matches, 2);
    for(unsigned i = 0; i < matches.size(); i++) {
        if(matches[i][0].distance < nn_match_ratio * matches[i][1].distance) {
            matched1.push_back(matches[i][0].queryIdx);
            matched2.push_back(matches[i][0].trainIdx);
        }
    }
    int num_matches = (int)matched1.size();
    if (num_matches < 4) return 0;

    Mat inlier_mask, homography;
    homography = findHomography(Points(matched1, prev_kp), Points(matched2, kp),
                                RANSAC, ransac_thresh, inlier_mask);

    for(unsigned i = 0; i < matched1.size(); i++) {
        if(inlier_mask.at<uchar>(i)) {
            out_inliers_prev.push_back(matched1[i]);
            out_inliers.push_back(matched2[i]);
        }
    }
    float inliers = (int)out_inliers_prev.size();
    float ratio = (num_matches!=0)? inliers * 1.0 / num_matches : 0;
    return ratio;
}

#endif // UTILS_H