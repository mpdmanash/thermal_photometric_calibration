
# Online Photometric Calibration of Automatic Gain Thermal Infrared Cameras
Thermal infrared cameras are increasingly being used in various applications such as robot vision, industrial inspection and medical imaging, thanks to their improved resolution and portability. However, the performance of traditional computer vision techniques developed for electro-optical imagery does not directly translate to the thermal domain due to two major reasons: these algorithms require photometric assumptions to hold, and methods for photometric calibration of RGB cameras cannot be applied to thermal-infrared cameras due to difference in data acquisition and sensor phenomenology. In this paper, we take a step in this direction, and introduce a novel algorithm for online photometric calibration of thermal-infrared cameras. Our proposed method does not require any specific driver/hardware support and hence can be applied to any commercial off-the-shelf thermal IR camera. We present this in the context of visual odometry and SLAM algorithms, and demonstrate the efficacy of our proposed system through extensive experiments for both standard benchmark datasets, and real-world field tests with a thermal-infrared camera in natural outdoor environments.

[**Check out our YouTube-Video, showing Online Thermal Photometric Calibration in Action**](https://youtu.be/364V3xBG1Tg)  
[![Joint Point Cloud and Image Based Localization For Efficient Inspection in Mixed Reality](https://img.youtube.com/vi/364V3xBG1Tg/0.jpg)](https://youtu.be/364V3xBG1Tg)

## Publication
If you use this code in an academic context, please cite the following [RAL-ICRA 2021 paper](https://ieeexplore.ieee.org/document/93611249).

Manash Pratim Das, Larry Matthies, Shreyansh Daftry: **Online Photometric Calibration of Automatic Gain Thermal Infrared Cameras**, IEEE Robotics and Automation Letters 6.2 (2021): 2453-2460.

```
@ARTICLE{9361124,
  author={Das, Manash Pratim and Matthies, Larry and Daftry, Shreyansh},
  journal={IEEE Robotics and Automation Letters}, 
  title={Online Photometric Calibration of Automatic Gain Thermal Infrared Cameras}, 
  year={2021},
  volume={6},
  number={2},
  pages={2453-2460},
  doi={10.1109/LRA.2021.3061401}}
```


Dependencies: [OpenCV (3.4.9)](https://github.com/opencv/opencv), [Ceres-Solver (1.13.0)](http://ceres-solver.org/), [Eigen (3.3.7)](https://eigen.tuxfamily.org/index.php?title=Main_Page) OpenMP, Boost Libraries 


Contact: Manash Pratim Das (mpdmansh @ cmu . com)
