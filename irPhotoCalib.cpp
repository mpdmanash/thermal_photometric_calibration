/*
 * Author: Manash Pratim Das (mpdmanash@iitkgp.ac.in) 
 */

#include "irPhotoCalib.h"

double get_wall_time(){
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        //  Handle error
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}
double get_cpu_time(){
    return (double)clock() / CLOCKS_PER_SEC;
}

class newDSinglePairGainAnaJCostFunc
  : public ceres::CostFunction {
 public:
  newDSinglePairGainAnaJCostFunc( vector<float> o, vector<float> op, int num_residuals)
                              : o(o), op(op) {
    set_num_residuals(num_residuals+2);
    mutable_parameter_block_sizes()->push_back(1);
    mutable_parameter_block_sizes()->push_back(1);
  }
  virtual ~newDSinglePairGainAnaJCostFunc() {}
  virtual bool Evaluate(double const* const* params,
                        double* residuals,
                        double** jacobians) const {
    double aip = *params[0];
    double bip = *params[1];
    double w = 0.1;
    
    for (int i=0; i<o.size(); i++){
      residuals[i] = double(o[i])-(double(op[i])*(aip-bip) + bip);
    }
    residuals[o.size()] = w*(aip - 1.0);
    residuals[o.size()+1] = w*(bip - 0.0);
    
    if (!jacobians) return true;

    if (jacobians[0] != NULL) // Jacobian for aip is requested
    {
      for (int k=0; k<o.size(); k++)
        jacobians[0][k] = -op[k];
      jacobians[0][o.size()] = w;
      jacobians[0][o.size()+1] = 0.0;
    }

    if (jacobians[1] != NULL) // Jacobian for bip is requested
    {
      for (int k=0; k<o.size(); k++)
        jacobians[1][k] = (op[k]-1.0);
      jacobians[1][o.size()] = 0.0;
      jacobians[1][o.size()+1] = w;
    }
    return true;
  }
  private:
  const vector<float> o, op;
};

class newSinglePairGainAnaJCostFunc
  : public ceres::SizedCostFunction<6, 1, 1> {
 public:
  newSinglePairGainAnaJCostFunc( vector<float> o, vector<float> op)
                              : o(o), op(op) {
  }
  virtual ~newSinglePairGainAnaJCostFunc() {}
  virtual bool Evaluate(double const* const* params,
                        double* residuals,
                        double** jacobians) const {
    double aip = params[0][0];
    double bip = params[1][0];
    double w = 0.1;
    
    for (int i=0; i<o.size(); i++){
      residuals[i] = double(o[i])-(double(op[i])*(aip-bip) + bip);
    }
    residuals[o.size()] = w*(aip - 1.0);
    residuals[o.size()+1] = w*(bip - 0.0);
    
    if (!jacobians) return true;

    if (jacobians[0] != NULL) // Jacobian for a is requested
    {
      for (int k=0; k<o.size(); k++)
        jacobians[0][k] = -op[k];
      jacobians[0][o.size()] = w;
      jacobians[0][o.size()+1] = 0.0;
    }

    if (jacobians[1] != NULL) // Jacobian for b is requested
    {
      for (int k=0; k<o.size(); k++)
        jacobians[1][k] = (op[k]-1.0);
      jacobians[1][o.size()] = 0.0;
      jacobians[1][o.size()+1] = w;
    }
    return true;
  }
  private:
  const vector<float> o, op;
};

IRPhotoCalib::IRPhotoCalib(int w, int h, int k_div, float k_calibrate_SP, float k_SP_threshold, bool useKeyframes)
{
  m_useKeyframes = useKeyframes;
  m_frame_id = 0;
  m_latest_KF_id = 0;
  PTAB first_frame_params; first_frame_params.a = 1.0; first_frame_params.b = 0.0;
  m_params_PT.push_back(first_frame_params);
  m_epsilon_gap = 0.1;
  m_epsilon_base = 0.4;
  m_div = k_div;
  m_w = w; m_h = h;

  // Spatial Params
  m_spatial_coverage = Mat(cv::Size(((m_h/m_div)) * ((m_w/m_div)),1), CV_32FC1, Scalar(0));
  m_calibrate_SP = k_calibrate_SP; m_SP_threshold = k_SP_threshold;
  m_SP_correscount = vector<int>((w*h)/(m_div*m_div),0);
  m_SP_max_correscount = 10;
}

IRPhotoCalib::~IRPhotoCalib()
{
}

PTAB IRPhotoCalib::getPrevAB()
{
  if (m_useKeyframes) return m_params_PT[m_latest_KF_id];
  else return m_params_PT[m_frame_id];
}

void IRPhotoCalib::getRelativeGains(double a1, double b1, double a2, double b2, double & a12, double & b12){
  double e12 = (a2-b2)/(a1-b1);
  b12 = (b2-b1)/(a1-b1);
  a12 = e12 + b12;
}

void IRPhotoCalib::chainGains(double a01, double b01, double a12, double b12, double & a02, double & b02){
  double e02 = (a01-b01) * (a12 - b12);
  b02 = b01+(a01-b01)*b12;
  a02 = e02 + b02;
}

int IRPhotoCalib::getNid(int ptx, int pty){
  return (int)(std::floor(pty/m_div) * std::floor(m_w/m_div) + std::floor(ptx/m_div));
}

PTAB IRPhotoCalib::ProcessCurrentFrame(vector<vector<float> > intensity_history, 
                                       vector<vector<float> > intensity_current, 
                                       vector<int> frame_ids_history, 
                                       vector<vector<pair<int,int> > > pixels_history,
                                       vector<vector<pair<int,int> > > pixels_current,
                                       bool thisKF)
{
  // Match this frame with previous frames
  PTAB prevAB = getPrevAB(); 
  double a_origin_previous = prevAB.a; double b_origin_previous = prevAB.b;
  double w_a = 0; double w_b = 0; int w_count = 0;
  for(int i=0; i<intensity_history.size(); i++){
      if (intensity_history[i].size()<=4) continue;
      vector<int> prev_inliers, inliers;
      double a_history_current, b_history_current, a_origin_current, b_origin_current, a_previous_current, b_previous_current;
      int support_points = EstimateGainsRansac(intensity_history[i], intensity_current[i], a_history_current, b_history_current);
      double a_origin_history = m_params_PT[m_frame_id+1-frame_ids_history[i]].a; double b_origin_history = m_params_PT[m_frame_id+1-frame_ids_history[i]].b;
      chainGains(a_origin_history, b_origin_history, a_history_current, b_history_current, a_origin_current, b_origin_current);
      getRelativeGains(a_origin_previous, b_origin_previous, a_origin_current, b_origin_current, a_previous_current, b_previous_current); // May be only do it previous key frame and not previour frame
      w_a += a_previous_current*support_points; w_b += b_previous_current*support_points; w_count += support_points;
  }
  double w_a_previous_current = w_a/w_count; double w_b_previous_current = w_b/w_count;
  if (w_count<5){w_a_previous_current=1.0; w_b_previous_current=0.0;} // in case, we do not have enough correspondence to estimate AB

  // Drift adjustment
  double delta = (1.0 - (w_a_previous_current-w_b_previous_current)) * m_epsilon_gap;
  w_a_previous_current = w_a_previous_current + delta;
  w_b_previous_current = w_b_previous_current - delta;
  w_a_previous_current = w_a_previous_current -(w_a_previous_current-1.0)*m_epsilon_base;
  w_b_previous_current = w_b_previous_current -(w_b_previous_current)*m_epsilon_base;

  double a_origin_current, b_origin_current;
  chainGains(a_origin_previous, b_origin_previous, w_a_previous_current, w_b_previous_current, a_origin_current, b_origin_current);

  // Spatial Calibration
  if(m_calibrate_SP){
    for(int i=0; i<pixels_current.size(); i++){
      double a_origin_history = m_params_PT[m_frame_id+1-frame_ids_history[i]].a; double b_origin_history = m_params_PT[m_frame_id+1-frame_ids_history[i]].b;
      double a_history_current, b_history_current;
      getRelativeGains(a_origin_history, b_origin_history, a_origin_current, b_origin_current, a_history_current, b_history_current);
      for(int j=0; j<pixels_current[i].size(); j++){
        int sid_history = this->getNid(pixels_history[i][j].first, pixels_history[i][j].second);
        int sid_current = this->getNid(pixels_current[i][j].first, pixels_current[i][j].second);
        if(sid_history==sid_current || (m_SP_correscount[sid_history] > m_SP_max_correscount && m_SP_correscount[sid_history] > m_SP_max_correscount)) continue;
        
        m_spatial_coverage.at<float>(0,sid_history) = 1;
        m_spatial_coverage.at<float>(0,sid_current) = 1;

        m_sids_history.push_back(sid_history);
        m_sids_current.push_back(sid_current);

        double bi = intensity_current[i][j]*(a_history_current-b_history_current) - intensity_history[i][j] + b_history_current;
        m_SP_vecB.push_back(bi);
        m_SP_correscount[sid_current]++; m_SP_correscount[sid_history]++;
      }
    }
    float coverage_ratio = cv::sum(m_spatial_coverage)[0]/(m_spatial_coverage.rows*m_spatial_coverage.cols);

    if(coverage_ratio > m_SP_threshold){
      m_calibrate_SP = false;
      std::async(std::launch::async, &IRPhotoCalib::EstimateSpatialParameters, this);
    }
  }

  m_frame_id++;
  if(thisKF) m_latest_KF_id = m_frame_id;
  PTAB this_frame_params; this_frame_params.a = a_origin_current; this_frame_params.b = b_origin_current;
  m_params_PT.push_back(this_frame_params);

  return this_frame_params;
}

int IRPhotoCalib::EstimateGainsRansac(vector<float> oi, vector<float> opip,
                                      double &out_aip, double &out_bip){
  vector<int> pickid;
  int count = 0;
  for (int i=0; i<oi.size(); i++) pickid.push_back(i);

  if (oi.size()<4){std::cout << oi.size() << " :Not enough Points for RANSAC\n"; return 0;}
  std::random_device rd;
  std::mt19937 g(rd());

  double best_aip, best_bip; vector<double> found_aips; vector<double> found_bips;
  int most_inliers = 0; vector<int> best_inliers; vector<int> best_outliers;
  for(int rsi=0; rsi<oi.size(); rsi++)
  {
    std::shuffle(pickid.begin(), pickid.end(), g);
    vector<float> this_o, this_op;
    this_o.push_back(oi[pickid[0]]);this_o.push_back(oi[pickid[1]]);this_o.push_back(oi[pickid[2]]);this_o.push_back(oi[pickid[3]]);
    this_op.push_back(opip[pickid[0]]);this_op.push_back(opip[pickid[1]]);this_op.push_back(opip[pickid[2]]);this_op.push_back(opip[pickid[3]]);
    vector<double> this_a(1), this_b(1);
    for (int i=0; i<1; i++){ this_a[i]=1.0; this_b[i]=0.0; }

    ceres::Problem problem;
    ceres::CostFunction* cost_function = new newSinglePairGainAnaJCostFunc(this_o, this_op);
    problem.AddResidualBlock(cost_function, NULL, &this_a[0], &this_b[0]);
    
    ceres::Solver::Options options;
    options.max_num_iterations = 50;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // Check RANSAC Votes with threshold
    double aip = this_a[0]; double bip = this_b[0];
    vector<int> inliers, outliers;
    double threshold = 8.0e-3;
    for(int i=0; i<oi.size(); i++)
    {
      double diff = fabs( double(oi[i])-(double(opip[i])*(aip-bip) + bip) );
      if (diff < threshold) inliers.push_back(i);
      else outliers.push_back(i);
    }
    found_aips.push_back(aip);
    found_bips.push_back(bip);

    if(inliers.size()>most_inliers)
    {
      most_inliers = inliers.size();
      best_aip = aip; best_bip = bip;
      best_inliers = inliers;
      best_outliers = outliers;
    }
  }

  // Estimate parameters based on inliers
  vector<float> inliers_o, inliers_op;
  for (int i=0; i<most_inliers; i++) {inliers_o.push_back(oi[best_inliers[i]]); inliers_op.push_back(opip[best_inliers[i]]);}
  vector<double> optimization_variables(2);
  optimization_variables[0]=1.0; optimization_variables[1]=0.0;
  vector<double*> parameter_blocks;
  for (int i = 0; i < optimization_variables.size(); ++i) {
    parameter_blocks.push_back(&(optimization_variables[i]));
  }
  ceres::Problem problem;
  problem.AddResidualBlock(new newDSinglePairGainAnaJCostFunc(inliers_o, inliers_op, most_inliers), NULL, parameter_blocks);
  ceres::Solver::Options options;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  out_aip = *parameter_blocks[0]; out_bip = *parameter_blocks[1];
  return most_inliers;
}

void IRPhotoCalib::EstimateSpatialParameters()
{
  cout << "ESTIMATING SP\n";
  vector<int> serial_to_variable_id((m_w*m_h)/(m_div*m_div),-1);
  vector<int> variable_to_serial_id;
  vector<pair<int, int> > Aposneg_id;

  for (int i=0; i<m_sids_history.size(); i++)
  {
    int sid_current = m_sids_current[i];
    int sid_history = m_sids_history[i];
    
    
    int vidp = -1; int vid = -1;
    if (serial_to_variable_id[sid_current]==-1)
    {
      // This variable is first. Thus add to serial_to_variable_id and variable_to_serial_id
      variable_to_serial_id.push_back(sid_current);
      int thisVid = variable_to_serial_id.size()-1;
      serial_to_variable_id[sid_current] = thisVid;
      vidp = thisVid;
    }
    else vidp = serial_to_variable_id[sid_current];

    if (serial_to_variable_id[sid_history]==-1)
    {
      // This variable is first. Thus add to serial_to_variable_id and variable_to_serial_id
      variable_to_serial_id.push_back(sid_history);
      int thisVid = variable_to_serial_id.size()-1;
      serial_to_variable_id[sid_history]= thisVid;
      vid = thisVid;
    }
    else vid = serial_to_variable_id[sid_history];

    Aposneg_id.push_back(make_pair(vidp,vid));
  }

  SpMat A(m_SP_vecB.size(), variable_to_serial_id.size());
  Eigen::MatrixXd b(m_SP_vecB.size(),1);
  std::vector<T> tripletList; tripletList.reserve(m_SP_vecB.size()*2);
  for (int i=0; i<m_SP_vecB.size(); i++)
  {
    b(i,0) = m_SP_vecB[i];
    tripletList.push_back(T(i,Aposneg_id[i].first,1.0));
    tripletList.push_back(T(i,Aposneg_id[i].second,-1.0));
  }
  A.setFromTriplets(tripletList.begin(), tripletList.end());
  
  // Eigen::SparseQR <SpMat, Eigen::COLAMDOrdering<int> > solver;
  Eigen::LeastSquaresConjugateGradient < SpMat > solver;

  solver.compute(A);
  if(solver.info() != Eigen::Success) {
    // decomposition failed
    cout << "failed\n";
    return;
  }
  Eigen::VectorXd x = solver.solve(b);
  if(solver.info() != Eigen::Success) {
    // solving failed
    cout << "failed 2\n";
    return;
  }
  m_calibrate_SP = false;
  cout << "Spatial Params Estimation Done\n";
}



float IRPhotoCalib::getCorrected(float o, int x, int y, vector< double >& a, vector< double >& b, vector< double >& s, vector< double >& n, int fid, int w, int h, int div, Mat& imNis, double minn, double maxn)
{
  double ai = a[fid]; double bi = b[fid];
  //double nxy = n[ getNid(x,y,div,w) ];
  double nxy = (imNis.at<uchar>( (int)std::floor(y/div) , (int)std::floor(x/div)) * (maxn-minn) / 255.0) + minn;
  float co = o*(ai-bi)+bi-nxy;
  //if (nxy == 0.000001) return 0.000001;
  return co;
}

float IRPhotoCalib::getGainCorrected(float o, int x, int y, vector< double >& a, vector< double >& b, vector< double >& s, vector< double >& n, int fid, int w, int h, int div)
{
  double ai = a[fid]; double bi = b[fid];
  float co = (o*exp(ai)+bi);
  return co;
}
