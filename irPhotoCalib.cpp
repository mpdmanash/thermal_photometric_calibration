/*
 * Author: Manash Pratim Das (mpdmanash@iitkgp.ac.in) 
 */

#include "irPhotoCalib.h"

static const int kStride = 10;

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

int getNid(int ptx, int pty, int div, int w)
{
  return (int)(std::floor(pty/div) * std::floor(w/div) + std::floor(ptx/div));
}

// Construct Non-Linear Error Function for cere-solver
struct PhotometricError 
{
  PhotometricError(vector<int> ptx, vector<int> pty, vector<int> pptx, vector<int> ppty, vector<float> o, vector<float> op, vector<int> id, vector<int> idp, int w, int h, int div, int n_frames)
      : ptx(ptx), pty(pty), pptx(pptx), ppty(ppty), o(o), op(op), id(id), idp(idp), w(w), h(h), div(div), n_frames(n_frames)  {}

  template <typename T>
  bool operator()(
          const T* const n,
          const T* const a,
          const T* const b,
                T* residuals) const {
    for(int i=0; i<ptx.size(); i++)
    {
      T t, tp;
      
      T ai = a[id[i]];
      T bi = b[id[i]];
      T aip = a[idp[i]];
      T bip = b[idp[i]];
      
      T nxy = T(n[ getNid(ptx[i], pty[i], div, w) ]);
      T nxyp = T(n[ getNid(pptx[i], ppty[i], div, w) ]);
      t = T(o[i])*ceres::exp(ai)+bi + nxy;
      tp = T(op[i])*ceres::exp(aip)+bip + nxyp;
      residuals[i] = t-tp;
      //residuals[i] = T(0);
    }
    for (int i=0; i<n_frames; i++)
    {
      T ai = a[i];
      T bi = b[i];
      residuals[ptx.size()+i] = 0.15*(ceres::abs(ai));
    }
    return true;
  }
private:

  const vector<int> ptx, pty, pptx, ppty, id, idp;
  const vector<float> o, op;
  const int w,h,div, n_frames;
};

// Construct Non-Linear Error Function for cere-solver
struct PhotometricErrorCMU 
{
  PhotometricErrorCMU(vector<int> ptx, vector<int> pty, vector<int> pptx, vector<int> ppty, vector<float> o, vector<float> op, vector<int> id, vector<int> idp, int w, int h, int div, int n_frames)
      : ptx(ptx), pty(pty), pptx(pptx), ppty(ppty), o(o), op(op), id(id), idp(idp), w(w), h(h), div(div), n_frames(n_frames)  {}

  template <typename T>
  bool operator()(
          const T* const n,
          const T* const a,
          const T* const b,
                T* residuals) const {
    for(int i=0; i<ptx.size(); i++)
    {
      T t, tp;      
      T ai = a[id[i]];
      T bi = b[id[i]];
      T aip = a[idp[i]];
      T bip = b[idp[i]];
      
      T nxy = T(n[ getNid(ptx[i], pty[i], div, w) ]);
      T nxyp = T(n[ getNid(pptx[i], ppty[i], div, w) ]);
      residuals[i] = T(o[i])-(T(op[i])*(aip-bip) + bip - nxyp - bi + nxy)/(ai-bi);
    }
    return true;
  }
private:

  const vector<int> ptx, pty, pptx, ppty, id, idp;
  const vector<float> o, op;
  const int w,h,div, n_frames;
};

class DSinglePairGainAnaJCostFunc
  : public ceres::CostFunction {
 public:
  DSinglePairGainAnaJCostFunc( vector<float> o, vector<float> op, int num_residuals)
                              : o(o), op(op) {
    set_num_residuals(num_residuals);
    mutable_parameter_block_sizes()->push_back(1);
    mutable_parameter_block_sizes()->push_back(1);
    mutable_parameter_block_sizes()->push_back(1);
    mutable_parameter_block_sizes()->push_back(1);
  }
  virtual ~DSinglePairGainAnaJCostFunc() {}
  virtual bool Evaluate(double const* const* params,
                        double* residuals,
                        double** jacobians) const {
    double ai = *params[0];
    double aip = *params[1];
    double bi = *params[2];
    double bip = *params[3];
    
    for (int i=0; i<o.size(); i++){
      residuals[i] = double(o[i])-(double(op[i])*(aip-bip) + bip - bi)/(ai-bi);
    }
    
    if (!jacobians) return true;

    if (jacobians[0] != NULL) // Jacobian for ai is requested
      for (int k=0; k<o.size(); k++)
        jacobians[0][k] = (op[k]*aip-op[k]*bip+bip-bi)/( (ai-bi)*(ai-bi) );

    if (jacobians[1] != NULL) // Jacobian for aip is requested
      for (int k=0; k<o.size(); k++)
        jacobians[1][k] = -op[k]/(ai-bi);

    if (jacobians[2] != NULL) // Jacobian for bi is requested
      for (int k=0; k<o.size(); k++)
        jacobians[2][k] = -(op[k]*aip-op[k]*bip+bip-bi)/( (ai-bi)*(ai-bi) );
    
    if (jacobians[3] != NULL) // Jacobian for bip is requested
      for (int k=0; k<o.size(); k++)
        jacobians[3][k] = (op[k]-1.0)/(ai-bi);
    return true;
  }
  private:
  const vector<float> o, op;
};

class PhotometricAnaJCostFunction
  : public ceres::CostFunction {
 public:
  PhotometricAnaJCostFunction(vector<int> ptx, vector<int> pty,
                              vector<int> pptx, vector<int> ppty,
                              vector<float> o, vector<float> op,
                              vector<int> id, vector<int> idp,
                              int w, int h, int div, int n_frames, int n_div, int i,
                              int num_residuals, int parameter_block_size)
                              : ptx(ptx), pty(pty), pptx(pptx), ppty(ppty), 
                                o(o), op(op), id(id), idp(idp), w(w), h(h), div(div),
                                n_frames(n_frames), n_div(n_div), i(i) {
    set_num_residuals(num_residuals);
    for(int i = 0; i<parameter_block_size; i++){mutable_parameter_block_sizes()->push_back(1);}
  }
  virtual ~PhotometricAnaJCostFunction() {}
  virtual bool Evaluate(double const* const* params,
                        double* residuals,
                        double** jacobians) const {
    for(int j=0; j<ptx.size(); j++)
    {  
      double ai = *params[n_div+id[j]];
      double bi = *params[n_div+n_frames+id[j]];
      double aip = *params[n_div+idp[j]];
      double bip = *params[n_div+n_frames+idp[j]];
      
      double nxy = *params[ getNid(ptx[j], pty[j], div, w) ];
      double nxyp = *params[ getNid(pptx[j], ppty[j], div, w) ];
      residuals[j] = double(o[j])-(double(op[j])*(aip-bip) + bip - nxyp - bi + nxy)/(ai-bi);
    }

    if (!jacobians) return true;

    for(int j=0; j<n_div; j++){ // For n's
      if (jacobians[j] != NULL){ // We need to give jacobian for param j but for all residuals
        for(int k=0; k<ptx.size(); k++ ){
          int n_num = getNid(ptx[k], pty[k], div, w);
          int np_num = getNid(pptx[k], ppty[k], div, w);
          int nxy_id = n_num;
          int nxyp_id = np_num;

          int f_num = id[k];
          double ai = *params[n_div+f_num];
          double bi = *params[n_div+n_frames+f_num];
          if (nxy_id == j) jacobians[j][k] = -1.0/(ai-bi);
          else if (nxyp_id == j) jacobians[j][k] = 1.0/(ai-bi);
          else jacobians[j][k] = 0.0;
        }
      }
    }

    for(int j=n_div; j<n_div+n_frames; j++){ // For a's
      if (jacobians[j] != NULL){ // We need to give jacobian for param j but for all residuals
        for(int k=0; k<ptx.size(); k++ ){
          int f_num = id[k];
          int fp_num = idp[k];
          int ai_id = n_div+f_num;
          int aip_id = n_div+fp_num;

          int n_num = getNid(ptx[k], pty[k], div, w);
          int np_num = getNid(pptx[k], ppty[k], div, w);
          double ai = *params[n_div+f_num];
          double bi = *params[n_div+n_frames+f_num];
          double aip = *params[n_div+fp_num];
          double bip = *params[n_div+n_frames+fp_num];
          double nxy = *params[n_num];
          double nxyp = *params[np_num];

          if (ai_id == j) jacobians[j][k] = (op[k]*aip-op[k]*bip+bip-nxyp-bi-nxy)/( (ai-bi)*(ai-bi) );
          else if (aip_id == j) jacobians[j][k] = -op[k]/(ai-bi);
          else jacobians[j][k] = 0.0;
        }
      }
    }

    for(int j=n_div+n_frames; j<n_div+2*n_frames; j++){ // For a's
      if (jacobians[j] != NULL){ // We need to give jacobian for param j but for all residuals
        for(int k=0; k<ptx.size(); k++ ){
          int f_num = id[k];
          int fp_num = idp[k];
          int bi_id = n_div+n_frames+f_num;
          int bip_id = n_div+n_frames+fp_num;

          int n_num = getNid(ptx[k], pty[k], div, w);
          int np_num = getNid(pptx[k], ppty[k], div, w);
          double ai = *params[n_div+f_num];
          double bi = *params[n_div+n_frames+f_num];
          double aip = *params[n_div+fp_num];
          double bip = *params[n_div+n_frames+fp_num];
          double nxy = *params[n_num];
          double nxyp = *params[np_num];

          if (bi_id == j) jacobians[j][k] = -(op[k]*aip-op[k]*bip+bip-nxyp-bi-nxy)/( (ai-bi)*(ai-bi) );
          else if (bip_id == j) jacobians[j][k] = (op[k]-1.0)/(ai-bi);
          else jacobians[j][k] = 0.0;
        }
      }
    }

    return true;
  }
  private:

  const vector<int> ptx, pty, pptx, ppty, id, idp;
  const vector<float> o, op;
  const int w,h,div, n_frames, n_div, i;
};

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
  : public ceres::SizedCostFunction<6 /* number of residuals */,
                             1 /* size of first parameter */,
                             1 /* size of second parameter */> {
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

class SinglePairGainAnaJCostFunc
  : public ceres::SizedCostFunction<4 /* number of residuals */,
                             2 /* size of first parameter */,
                             2 /* size of second parameter */> {
 public:
  SinglePairGainAnaJCostFunc( vector<float> o, vector<float> op)
                              : o(o), op(op) {
  }
  virtual ~SinglePairGainAnaJCostFunc() {}
  virtual bool Evaluate(double const* const* params,
                        double* residuals,
                        double** jacobians) const {
    double ai = params[0][0];
    double aip = params[0][1];
    double bi = params[1][0];
    double bip = params[1][1];
    
    for (int i=0; i<o.size(); i++){
      residuals[i] = double(o[i])-(double(op[i])*(aip-bip) + bip - bi)/(ai-bi);
    }
    
    if (!jacobians) return true;

    if (jacobians[0] != NULL) // Jacobian for a is requested
    {
      for (int k=0; k<o.size(); k++)
      {
        jacobians[0][k*2] = (op[k]*aip-op[k]*bip+bip-bi)/( (ai-bi)*(ai-bi) );
        jacobians[0][k*2+1] = -op[k]/(ai-bi);
      }
    }

    if (jacobians[1] != NULL) // Jacobian for b is requested
    {
      for (int k=0; k<o.size(); k++)
      {
        jacobians[1][k*2] = -(op[k]*aip-op[k]*bip+bip-bi)/( (ai-bi)*(ai-bi) );
        jacobians[1][k*2+1] = (op[k]-1.0)/(ai-bi);
      }
    }    

    return true;
  }
  private:
  const vector<float> o, op;
};

struct RangeConstraint {
  typedef ceres::DynamicAutoDiffCostFunction<RangeConstraint, kStride>
      RangeCostFunction;
  RangeConstraint(vector<int> ptx, vector<int> pty, vector<int> pptx, vector<int> ppty, vector<float> o, vector<float> op, vector<int> id, vector<int> idp, int w, int h, int div, int n_frames, int n_div, int i) :
      ptx(ptx), pty(pty), pptx(pptx), ppty(ppty), o(o), op(op), id(id), idp(idp), w(w), h(h), div(div), n_frames(n_frames), n_div(n_div), i(i) {}
  template <typename T>
  bool operator()(T const* const* params, T* residuals) const {
    for(int j=0; j<ptx.size(); j++)
    {  
      T ai = *params[n_div+id[j]];
      T bi = *params[n_div+n_frames+id[j]];
      T aip = *params[n_div+idp[j]];
      T bip = *params[n_div+n_frames+idp[j]];
      
      T nxy = *params[ getNid(ptx[j], pty[j], div, w) ];
      T nxyp = *params[ getNid(pptx[j], ppty[j], div, w) ];
      residuals[j] = T(o[j])-(T(op[j])*(aip-bip) + bip - nxyp - bi + nxy)/(ai-bi);
    }
    return true;
  }
  // Factory method to create a CostFunction from a RangeConstraint to
  // conveniently add to a ceres problem.
  static RangeCostFunction* Create(const vector<int> ptx, const vector<int> pty,
                                   const vector<int> pptx, const vector<int> ppty,
                                   const vector<float> o, const vector<float> op, 
                                   const vector<int> id, const vector<int> idp, 
                                   const int w, const int h, const int div, const int n_frames, const int n_div, const int i,
                                   vector<double>* optimization_variables,
                                   vector<double*>* parameter_blocks) {
    RangeConstraint* constraint = new RangeConstraint(
        ptx,pty,pptx,ppty,o,op,id,idp,w,h,div,n_frames,n_div,i);
    RangeCostFunction* cost_function = new RangeCostFunction(constraint);
    // Add all the parameter blocks that affect this constraint.
    parameter_blocks->clear();
    for (int i = 0; i <= optimization_variables->size(); ++i) {
      parameter_blocks->push_back(&((*optimization_variables)[i]));
      cost_function->AddParameterBlock(1);
    }
    cost_function->SetNumResiduals(ptx.size());
    return (cost_function);
  }
  const vector<int> ptx, pty, pptx, ppty, id, idp;
  const vector<float> o, op;
  const int w,h,div, n_frames, n_div, i;
};

// struct RangeConstraintSingleResidual {
//   typedef ceres::DynamicAutoDiffCostFunction<RangeConstraintSingleResidual, kStride>
//       RangeCostFunction;
//   RangeConstraintSingleResidual(vector<int> ptx, vector<int> pty, vector<int> pptx, vector<int> ppty, vector<float> o, vector<float> op, vector<int> id, vector<int> idp, int w, int h, int div, int n_frames, int n_div, int i) :
//       ptx(ptx), pty(pty), pptx(pptx), ppty(ppty), o(o), op(op), id(id), idp(idp), w(w), h(h), div(div), n_frames(n_frames), n_div(n_div), i(i) {}
//   template <typename T>
//   bool operator()(T const* const* params, T* residuals) const {
//     T t, tp;      
//     T ai = *params[n_div+id[i]];
//     T bi = *params[n_div+n_frames+id[i]];
//     T aip = *params[n_div+idp[i]];
//     T bip = *params[n_div+n_frames+idp[i]];
    
//     T nxy = *params[ getNid(ptx[i], pty[i], div, w) ];
//     T nxyp = *params[ getNid(pptx[i], ppty[i], div, w) ];
//     residuals[0] = T(o[i])-(T(op[i])*(aip-bip) + bip - nxyp - bi + nxy)/(ai-bi);
//     return true;
//   }
//   // Factory method to create a CostFunction from a RangeConstraintSingleResidual to
//   // conveniently add to a ceres problem.
//   static RangeCostFunction* Create(const vector<int> ptx, const vector<int> pty,
//                                    const vector<int> pptx, const vector<int> ppty,
//                                    const vector<float> o, const vector<float> op, 
//                                    const vector<int> id, const vector<int> idp, 
//                                    const int w, const int h, const int div, const int n_frames, const int n_div, const int i,
//                                    vector<double>* optimization_variables,
//                                    vector<double*>* parameter_blocks) {
//     RangeConstraintSingleResidual* constraint = new RangeConstraintSingleResidual(
//         ptx,pty,pptx,ppty,o,op,id,idp,w,h,div,n_frames,n_div,i);
//     RangeCostFunction* cost_function = new RangeCostFunction(constraint);
//     // Add all the parameter blocks that affect this constraint.
//     // parameter_blocks->clear();
//     for (int i = 0; i <= optimization_variables->size(); ++i) {
//     //   parameter_blocks->push_back(&((*optimization_variables)[i]));
//        cost_function->AddParameterBlock(1);
//     }
//     //cost_function->AddParameterBlock(n_div+2*n_frames);
//     cost_function->SetNumResiduals(1);
//     return (cost_function);
//   }
//   const vector<int> ptx, pty, pptx, ppty, id, idp;
//   const vector<float> o, op;
//   const int w,h,div, n_frames, n_div, i;
// };

struct GainError 
{
  GainError(vector<int> ptx, vector<int> pty, vector<int> pptx, vector<int> ppty, vector<float> o, vector<float> op, vector<int> id, vector<int> idp, int w, int h, int div, int n_frames, vector<double> n)
      : ptx(ptx), pty(pty), pptx(pptx), ppty(ppty), o(o), op(op), id(id), idp(idp), w(w), h(h), div(div), n_frames(n_frames), n(n)  {}

  template <typename T>
  bool operator()(
          const T* const a,
          const T* const b,
                T* residuals) const {
    for(int i=0; i<ptx.size(); i++)
    {
      T t, tp;
      
      T ai = a[id[i]];
      T bi = b[id[i]];
      T aip = a[idp[i]];
      T bip = b[idp[i]];
      
      T nxy = T(n[ getNid(ptx[i], pty[i], div, w) ]);
      T nxyp = T(n[ getNid(pptx[i], ppty[i], div, w) ]);
      
      t = (T(o[i])*ceres::exp(ai)+bi) + nxy;
      tp = (T(op[i])*ceres::exp(aip)+bip) + nxyp;
      residuals[i] = t-tp;
      //residuals[i] = T(0);
    }
    for (int i=0; i<n_frames; i++)
    {
      T ai = a[i];
      T bi = b[i];
      residuals[ptx.size()+i] = 0.15*(ceres::abs(ai));
    }
    return true;
  }
private:

  const vector<int> ptx, pty, pptx, ppty, id, idp;
  const vector<float> o, op;
  vector<double> n;
  const int w,h,div, n_frames;
};

struct NewPhotometricErrorPull 
{
  NewPhotometricErrorPull(int i)
      : i(i)  {}

  template <typename T>
  bool operator()(
          const T* const a,
                T* residuals) const {
      T ai = a[i];
      residuals[0] = 0.15*(ceres::abs(ai));    
    return true;
  }
private:
  const int i;
};

struct NewPhotometricError 
{
  NewPhotometricError(float o, float op, int id, int idp, int idn, int idnp)
      : o(o), op(op), id(id), idp(idp), idn(idn), idnp(idnp)  {}

  template <typename T>
  bool operator()(
          const T* const n,
          const T* const a,
          const T* const b,
                T* residuals) const {
      T t, tp;
      T ai = a[id];
      T bi = b[id];
      T aip = a[idp];
      T bip = b[idp];
      
      T nxy = n[  idn   ];
      T nxyp = n[  idnp  ];
      t = (T(o)*ceres::exp(ai)+bi) + nxy;
      tp = (T(op)*ceres::exp(aip)+bip) + nxyp;
      residuals[0] = t-tp;
    
    return true;
  }
private:

  const float o, op;
  const int id, idp, idn, idnp;
};

struct NewGainError 
{
  NewGainError(float o, float op, int id, int idp, double nxy, double nxyp)
      : o(o), op(op), id(id), idp(idp), nxy(nxy), nxyp(nxyp)  {}

  template <typename T>
  bool operator()(
          const T* const a,
          const T* const b,
                T* residuals) const {
      T t, tp;
      T ai = a[id];
      T bi = b[id];
      T aip = a[idp];
      T bip = b[idp];
      t = (T(o)*ceres::exp(ai)+bi) + T(nxy);
      tp = (T(op)*ceres::exp(aip)+bip) + T(nxyp);
      residuals[0] = t-tp;
    
    return true;
  }
private:

  const float o, op;
  const int id, idp;
  const double nxy, nxyp;
};

IRPhotoCalib::IRPhotoCalib()
{
  e_photo_error = 0.00019142419;
}

IRPhotoCalib::~IRPhotoCalib()
{
}

class CustomCallback : public ceres::IterationCallback {
 public:
  explicit CustomCallback(double threshold)
  : threshold(threshold){}

  ~CustomCallback() {}

  ceres::CallbackReturnType operator()(const ceres::IterationSummary& summary) {
    if(summary.cost < threshold)
      return ceres::SOLVER_TERMINATE_SUCCESSFULLY;
    else
      return ceres::SOLVER_CONTINUE;
  }

 private:
  const double threshold;
};

void IRPhotoCalib::RunOptimizer(vector<int> ptx, vector<int> pty, 
                            vector<int> pptx, vector<int> ppty, 
                            vector<float> o, vector<float> op, 
                            vector<int> id, vector<int> idp, 
                            int w, int h, int div, int n_frames,
                            vector<double> &out_s, vector<double> &out_n,
                            vector<double> &out_a, vector<double> &out_b)
{
  int n_pairs = ptx.size();
  int n_div = w*h/(div*div);
  vector<double> n(n_div); vector<double> a(n_frames); vector<double> b(n_frames);
  for (int i=0; i<n_div; i++)
  { n[i] = 0.000001; }
  for (int i=0; i<n_frames; i++)
  { a[i] = 0; b[i] = 0; }
  cout << "Making the problen\n";
  ceres::Problem problem;
  problem.AddResidualBlock(
    new ceres::AutoDiffCostFunction<PhotometricError, ceres::DYNAMIC, 2120, 30, 30>(
            new PhotometricError(ptx, pty, pptx, ppty, o, op, id, idp, w, h, div, n_frames),
            n_pairs+n_frames), NULL,
    &n[0],&a[0],&b[0]
  );

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
  options.minimizer_progress_to_stdout = true;
  options.max_num_iterations = 6;
  options.num_threads = 8;
  options.num_linear_solver_threads = 8;
  options.function_tolerance = 1e-5;
  CustomCallback ccb(this->e_photo_error*n_pairs);
  options.callbacks.push_back(&ccb);
  //options.dense_linear_algebra_library_type = ceres::LAPACK;
  //options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
  ceres::Solver::Summary summary;
  cout << "Calling the solver\n";
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.FullReport() << "\n";
  
  out_a = a; out_b = b; out_n = n;
  //out_a.push_back(0.0); out_b.push_back(0.0);
}

int IRPhotoCalib::RansacGains(vector<int> ptx, vector<int> pty, 
                            vector<int> pptx, vector<int> ppty, 
                            vector<float> o, vector<float> op, 
                            vector<int> id, vector<int> idp, 
                            int w, int h, int div,
                            int desired_i, int desired_ip, int offset, string dir,
                            double &out_aip, double &out_bip, vector<double> &mean_devs)
{
  vector<int> ptxi, ptyi, pptxip, pptyip, pickid;
  vector<float> oi, opip;
  int count = 0;
  for (int i=0; i<ptx.size(); i++)
  {
    if (id[i]==desired_i && idp[i]==desired_ip)
    {
      ptxi.push_back(ptx[i]); pptxip.push_back(pptx[i]);
      ptyi.push_back(pty[i]); pptyip.push_back(ppty[i]);
      oi.push_back(o[i]); opip.push_back(op[i]);
      pickid.push_back(count);
      count++;
    }
  }

  if (ptxi.size()<4){std::cout << ptxi.size() << " :Not enough Points for RANSAC\n"; return 0;}
  //else{std::cout << "Running RANSAC with " << ptxi.size() << " points\n";}
  std::random_device rd;
  std::mt19937 g(rd());

  double best_aip, best_bip; vector<double> found_aips; vector<double> found_bips;
  int most_inliers = 0; vector<int> best_inliers; vector<int> best_outliers;
  for(int rsi=0; rsi<ptxi.size()*4; rsi++)
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
    // for (int j=0; j<2; j++){
    //   problem.SetParameterLowerBound(&this_a[0], j, 0.5);
    //   problem.SetParameterUpperBound(&this_a[0], j, 1.0);
    //   problem.SetParameterLowerBound(&this_b[0], j, 0.0);
    //   problem.SetParameterUpperBound(&this_b[0], j, 0.5);
    // }
    // Run the solver!
    ceres::Solver::Options options;
    //options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 50;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    //std::cout << summary.BriefReport() << "\n";

    // Check RANSAC Votes with threshold
    double aip = this_a[0]; double bip = this_b[0];
    vector<int> inliers, outliers;
    double threshold = 8.0e-3;
    for(int i=0; i<ptxi.size(); i++)
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
  double mean_aip=0.0; double mean_bip=0.0;
  for (int i=0; i<found_aips.size(); i++){mean_aip+=found_aips[i]; mean_bip+=found_bips[i];}
  double std_dev_aip=0.0; double std_dev_bip=0.0;
  for (int i=0; i<found_aips.size(); i++){std_dev_aip+=std::pow(found_aips[i]-mean_aip,2); std_dev_bip+=std::pow(found_bips[i]-mean_bip,2);}
  std_dev_aip = std::sqrt(std_dev_aip); std_dev_bip = std::sqrt(std_dev_bip);
  mean_devs.clear();
  mean_devs.push_back(mean_aip); mean_devs.push_back(std_dev_aip);
  mean_devs.push_back(mean_bip); mean_devs.push_back(std_dev_bip);

  // RANSAC solved
  //std::cout << best_aip <<" " << best_bip << " : " << most_inliers << std::endl;
  vector<float> inliers_o, inliers_op;
  for (int i=0; i<most_inliers; i++)
  {
    inliers_o.push_back(oi[best_inliers[i]]); inliers_op.push_back(opip[best_inliers[i]]);
  }
  vector<double> optimization_variables(2);
  optimization_variables[0]=1.0; optimization_variables[1]=0.0;
  vector<double*> parameter_blocks;
  for (int i = 0; i < optimization_variables.size(); ++i) {
    parameter_blocks.push_back(&(optimization_variables[i]));
  }
  ceres::Problem problem;
  problem.AddResidualBlock(new newDSinglePairGainAnaJCostFunc(inliers_o, inliers_op, most_inliers), NULL, parameter_blocks);
  // for (int j=0; j<4; j++){
  //   problem.SetParameterLowerBound(parameter_blocks[j], 0, 0.0);
  //   problem.SetParameterUpperBound(parameter_blocks[j], 0, 1.0);
  // }
  // Run the solver!
  ceres::Solver::Options options;
  //options.minimizer_progress_to_stdout = true;
  //options.max_num_iterations = 10;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  //std::cout << summary.FullReport() << "\n";
  //std::cout << *parameter_blocks[0] <<" " << *parameter_blocks[1] << " : " << most_inliers << std::endl;

  out_aip = *parameter_blocks[0]; out_bip = *parameter_blocks[1];
  // double aip = *parameter_blocks[0]; double bip = *parameter_blocks[1];

  // stringstream hims_name, CMhims_name, imnew_name, imold_name;
  // hims_name << dir << "himnew" << desired_ip+offset << "imold" << desired_i+offset << ".png";
  // CMhims_name << dir << "CMhimnew" << desired_ip+offset << "imold" << desired_i+offset << ".png";
  // imnew_name << dir << "imnew" << desired_ip+offset << ".png";
  // imold_name << dir << "imold" << desired_i+offset << ".png";

  // Mat imnew = imread(imnew_name.str(), CV_LOAD_IMAGE_COLOR);
  // Mat imold = imread(imold_name.str(), CV_LOAD_IMAGE_COLOR);
  // Mat Cimnew; imnew.copyTo(Cimnew);
  // static Mat imouliers(imnew.rows, imnew.cols, CV_8UC3, Scalar(255,255,255));
  // static Mat iminliers(imnew.rows, imnew.cols, CV_8UC3, Scalar(255,255,255));
  // for(int r=0; r<imnew.rows; r++)
  // {
  //   for(int c=0; c<imnew.cols; c++)
  //   {
  //     float op = imnew.at<Vec3b>(r,c)[0]/255.0;
  //     float cop = op*(aip-bip)+bip;
  //     if (cop>1.0 && cop<0.0) std::cout << r<<" "<<c<<std::endl;
  //     if(cop>1.0) cop = 1.0;
  //     else if(cop<0.0) cop = 0.0;
  //     Cimnew.at<Vec3b>(r,c)[0] = (int)(cop*255.0);
  //     Cimnew.at<Vec3b>(r,c)[1] = (int)(cop*255.0);
  //     Cimnew.at<Vec3b>(r,c)[2] = (int)(cop*255.0);
  //   }
  // }
  // RNG rng(12345);
  // for(int i=0; i<best_inliers.size(); i++)
  // {
  //   Scalar color = Scalar(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255));
  //   circle(imold, Point(ptxi[best_inliers[i]], ptyi[best_inliers[i]]), 6, color, -1, 8, 0);
  //   circle(Cimnew, Point(pptxip[best_inliers[i]], pptyip[best_inliers[i]]), 6, color, -1, 8, 0);
  //   circle(iminliers, Point(pptxip[best_inliers[i]], pptyip[best_inliers[i]]), 1, color, -1, 8, 0);
  // }
  // for(int i=0; i<best_outliers.size(); i++)
  // {
  //   Scalar color = Scalar(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255));
  //   circle(imold, Point(ptxi[best_outliers[i]], ptyi[best_outliers[i]]), 12, color, 3, 8, 0);
  //   circle(Cimnew, Point(pptxip[best_outliers[i]], pptyip[best_outliers[i]]), 12, color, 3, 8, 0);
  //   circle(imouliers, Point(pptxip[best_outliers[i]], pptyip[best_outliers[i]]), 1, color, -1, 8, 0);
  // }
  // Mat hims, CMhims, holdims, all_ims;
  // hconcat(imold, Cimnew, hims);
  // hconcat(imold, imnew, holdims);
  // vconcat(hims, holdims, all_ims);
  // imwrite(hims_name.str(), all_ims);
  // Mat Call_ims;
  // applyColorMap(all_ims, Call_ims, COLORMAP_JET);
  // imwrite(CMhims_name.str(), Call_ims);
  // imwrite("outliers_map.png",imouliers);
  // imwrite("inliers_map.png",iminliers);
  return ptxi.size();
}

///// Trying to use 1 cost function with n_pairs residuals. Works.
void IRPhotoCalib::RunOptimizerCMU(vector<int> ptx, vector<int> pty, 
                            vector<int> pptx, vector<int> ppty, 
                            vector<float> o, vector<float> op, 
                            vector<int> id, vector<int> idp, 
                            int w, int h, int div, int n_frames,
                            vector<double> &out_s, vector<double> &out_n,
                            vector<double> &out_a, vector<double> &out_b)
{
  int n_pairs = ptx.size();
  int n_div = w*h/(div*div);
  std::cout << n_div << " N Div " << n_pairs << " N pairs\n";
  vector<double> n(n_div); vector<double> a(n_frames); vector<double> b(n_frames);
  vector<double> optimization_variables(n_div+2*n_frames);
   
  for (int i=0; i<n_div; i++)
  { n[i] = 0.000001; optimization_variables[i]=0.000001; }
  for (int i=0; i<n_frames; i++)
  { a[i] = 1; b[i] = 0; optimization_variables[n_div+i]=1.0; optimization_variables[n_div+n_frames+i]=0.0; }
  if (out_a.size()==n_frames && out_b.size()==n_frames){
    for (int i=0; i<n_frames; i++)
    { optimization_variables[n_div+i]=out_a[i]; optimization_variables[n_div+n_frames+i]=out_b[i]; }
  }

  cout << "Making the problen\n";
  ceres::Problem problem;
 
  vector<double*> parameter_blocks;
  for (int i = 0; i < optimization_variables.size(); ++i) {
    parameter_blocks.push_back(&(optimization_variables[i]));
  }
  // RangeConstraint::RangeCostFunction* range_cost_function =
  //     RangeConstraint::Create(
  //         ptx,pty,pptx,ppty,o,op,id,idp,w,h,div,n_frames,n_div,0, &optimization_variables, &parameter_blocks);
  // problem.AddResidualBlock(range_cost_function, NULL, parameter_blocks);
  problem.AddResidualBlock(new PhotometricAnaJCostFunction(ptx,pty,pptx,ppty,o,op,id,idp,w,h,div,n_frames,n_div,0,n_pairs,optimization_variables.size()),
                           NULL, parameter_blocks);

  for (int j=0; j<n_frames; j++){
    problem.SetParameterLowerBound(parameter_blocks[j+n_div], 0, 0.0);
    problem.SetParameterUpperBound(parameter_blocks[j+n_div], 0, 1.0);
    problem.SetParameterLowerBound(parameter_blocks[j+n_div+n_frames], 0, 0.0);
    problem.SetParameterUpperBound(parameter_blocks[j+n_div+n_frames], 0, 1.0);
  }
  // for (int j=0; j<n_div; j++){
  //   problem.SetParameterLowerBound(parameter_blocks[j], 0, 0.0);
  // }

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  options.minimizer_progress_to_stdout = true;
  options.max_num_iterations = 50;
  options.num_threads = 8;
  options.num_linear_solver_threads = 8;
  options.function_tolerance = 1e-5;
  ceres::Solver::Summary summary;
  cout << "Calling the solver\n";
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.FullReport() << "\n";

  for (int i=0; i<n_div; i++)
  { n[i] = optimization_variables[i]; }
  for (int i=0; i<n_frames; i++)
  { a[i] = optimization_variables[n_div+i]; b[i] = optimization_variables[n_div+n_frames+i]; }
  
  out_a = a; out_b = b; out_n = n;
}


/*///// Trying to use 1 cost function per data point. Does not work.
void IRPhotoCalib::RunOptimizerCMU(vector<int> ptx, vector<int> pty, 
                            vector<int> pptx, vector<int> ppty, 
                            vector<float> o, vector<float> op, 
                            vector<int> id, vector<int> idp, 
                            int w, int h, int div, int n_frames,
                            vector<double> &out_s, vector<double> &out_n,
                            vector<double> &out_a, vector<double> &out_b)
{
  int n_pairs = ptx.size();
  int n_div = w*h/(div*div);
  std::cout << n_div << " N Div " << n_pairs << " N pairs\n";
  vector<double> n(n_div); vector<double> a(n_frames); vector<double> b(n_frames);
  vector<double> optimization_variables(n_div+2*n_frames);
  for (int i=0; i<n_div; i++)
  { n[i] = 0.000001; optimization_variables[i]=0.000001; }
  for (int i=0; i<n_frames; i++)
  { a[i] = 1; b[i] = 0; optimization_variables[n_div+i]=1.0; optimization_variables[n_div+n_frames+i]=0.0; }

  cout << "Making the problen\n";
  ceres::Problem problem;
  vector<double*> parameter_blocks;
  for (int i = 0; i <= optimization_variables.size(); ++i) {
    parameter_blocks.push_back(&(optimization_variables[i]));
  }
  for (int i = 0; i < n_pairs; ++i) {
    RangeConstraintSingleResidual::RangeCostFunction* range_cost_function =
        RangeConstraintSingleResidual::Create(
            ptx,pty,pptx,ppty,o,op,id,idp,w,h,div,n_frames,n_div,i, &optimization_variables, &parameter_blocks);
    problem.AddResidualBlock(range_cost_function, NULL, parameter_blocks);
  }
  for (int j=0; j<n_frames; j++){
    problem.SetParameterLowerBound(parameter_blocks[j+n_div], 0, 0.0);
    problem.SetParameterUpperBound(parameter_blocks[j+n_div], 0, 1.0);
    problem.SetParameterLowerBound(parameter_blocks[j+n_div+n_frames], 0, 0.0);
    problem.SetParameterUpperBound(parameter_blocks[j+n_div+n_frames], 0, 1.0);
  }

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  options.minimizer_progress_to_stdout = true;
  options.max_num_iterations = 6;
  options.num_threads = 8;
  options.num_linear_solver_threads = 8;
  options.function_tolerance = 1e-5;
  ceres::Solver::Summary summary;
  cout << "Calling the solver\n";
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.FullReport() << "\n";

  for (int i=0; i<n_div; i++)
  { n[i] = optimization_variables[i]; }
  for (int i=0; i<n_frames; i++)
  { a[i] = optimization_variables[n_div+i]; b[i] = optimization_variables[n_div+n_frames+i]; }
  
  out_a = a; out_b = b; out_n = n;
}*/


void IRPhotoCalib::RunGainOptimizer(vector<int> ptx, vector<int> pty, 
                            vector<int> pptx, vector<int> ppty, 
                            vector<float> o, vector<float> op, 
                            vector<int> id, vector<int> idp, 
                            int w, int h, int div, int n_frames,
                            vector<double> &n,
                            vector<double> &out_a, vector<double> &out_b)
{
  int n_pairs = ptx.size();
  vector<double> a(n_frames); vector<double> b(n_frames);
  for (int i=0; i<n_frames; i++)
  { a[i] = 0; b[i] = 0; }
  cout << "Making the problen\n";
  ceres::Problem problem;
  problem.AddResidualBlock(
    new ceres::AutoDiffCostFunction<GainError, ceres::DYNAMIC, 30, 30>(
            new GainError(ptx, pty, pptx, ppty, o, op, id, idp, w, h, div, n_frames, n),
            n_pairs+n_frames), NULL,
    &a[0],&b[0]
  );
  //problem.SetParameterBlockConstant(&a[0]);
  //problem.SetParameterBlockConstant(&n[0]);

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
  options.minimizer_progress_to_stdout = true;
  //options.max_num_iterations = 16;
  options.num_threads = 8;
  options.num_linear_solver_threads = 8;
  options.function_tolerance = 1e-5;
  CustomCallback ccb(this->e_photo_error*n_pairs);
  options.callbacks.push_back(&ccb);
  ceres::Solver::Summary summary;
  cout << "Calling the solver\n";
  ceres::Solve(options, &problem, &summary);
  
  out_a = a; out_b = b;
  //out_a.push_back(0.0); out_b.push_back(0.0);
}

void IRPhotoCalib::RunGNAOptimizer(vector<int> ptx, vector<int> pty, 
                            vector<int> pptx, vector<int> ppty, 
                            vector<float> o, vector<float> op, 
                            vector<int> id, vector<int> idp, 
                            int w, int h, int div, int n_frames,
                            vector<double> &out_s, vector<double> &out_n,
                            vector<double> &out_a, vector<double> &out_b)
{
    int n_div = w*h/(div*div);
    vector<double> n(n_div); vector<double> a(n_frames); vector<double> b(n_frames);
    for (int i=0; i<n_div; i++)
    { n[i] = 0.0; }
    for (int i=0; i<n_frames; i++)
    { a[i] = 1.0; b[i] = 0.0; }
    
    Eigen::VectorXd dX;
    std::cout << "Starting Optimizer\n";
    for (int i=0; i<3; i++)
    {
        std::cout << "Optimizer Loop Start\n";
        //Linearize the system
        Eigen::MatrixXd B;
        SpMat Hs;
        vector<int> nz_rows;
        LinearizeSystem(ptx, pty, pptx, ppty, o, op, id, idp, w, h, div, n_frames, n, a, b, B, Hs, nz_rows);
        double t_op_start = get_wall_time();
        //COLAMD ordering
        
        Hs.makeCompressed();
        Eigen::BiCGSTAB<SpMat,Eigen::IncompleteLUT<double> > solver;
        solver.setMaxIterations(5);
        //solver.analyzePattern(Hs);
        solver.preconditioner().compute(Hs);
        solver.compute(Hs);
        if(solver.info()!=Eigen::Success) {
            std::cout << "failed 2\n";
            return;
        }
        for(int ms = 0; ms<1; ms++){
            dX = solver.solve(B);
            std::cout << "#iterations:     " << solver.iterations() << std::endl;
            std::cout << "estimated error: " << solver.error()      << std::endl;
        }
        
        /*Eigen::ConjugateGradient <SpMat, Eigen::Upper, Eigen::DiagonalPreconditioner<double> > solver;
        solver.setMaxIterations(200);
        solver.compute(Hs);
        if(solver.info()!=Eigen::Success) {
            // decomposition failed
            cout << "failed in compute ------------ !!!\n";
        }
        dX = solver.solve(B);
        if(solver.info()!=Eigen::Success) {
            // decomposition failed
            cout << "failed in solve ------------ !!!\n";
        }
        std::cout << "#iterations:     " << solver.iterations() << std::endl;
        std::cout << "estimated error: " << solver.error()      << std::endl;*/
        
        double t_op_end = get_wall_time();
        std::cout << t_op_end-t_op_start << " solved in time\n";
        for(int j=0; j<dX.size(); j++)
        {
            int o_row = nz_rows[j];
            if(o_row < n_div)
                n[o_row] = n[o_row]-dX(j,0);
            if(o_row >= n_div && o_row<n_div+n_frames)
                a[o_row-n_div] = a[o_row-n_div]-dX(j,0);
            if(o_row >= n_div+n_frames)
                b[o_row-n_div-n_frames] = b[o_row-n_div-n_frames]-dX(j,0);
        }
    }
    out_a = a; out_b = b; out_n = n;
    for (int i=0; i<n_frames; i++)
    { out_a[i] = out_a[i]; out_b[i] = out_b[i]; }
}

void IRPhotoCalib::LinearizeSystem(vector<int> &ptx, vector<int>& pty, 
                            vector<int>& pptx, vector<int>& ppty, 
                            vector<float>& o, vector<float>& op, 
                            vector<int>& id, vector<int>& idp, 
                            int w, int h, int div, int n_frames,
                      vector<double> n, vector<double> a, vector<double> b,
                      Eigen::MatrixXd &B, SpMat &Hs, vector<int> &nz_rows)
{
    double lambda = 1.5;
    std::cout << "Linearize System Start\n";
    Eigen::MatrixXd R;
    GetResiduals(ptx, pty, pptx, ppty, o, op, id, idp, w, h, div, n_frames, n, a, b, R);
    
    /*GetAnaJacobian(ptx, pty, pptx, ppty, o, op, id, idp, w, h, div, n_frames, n, a, b, J);
    Eigen::MatrixXd Jt = J.transpose();
    std::cout << "Taken Transpose\n";
    std::cout << "Size" << Jt.rows() << " " << Jt.cols() << "\n";
    H = Jt*J;*/
    
    SpMat J, Jt;
    int n_div = w*h/(div*div);
    int num_vars = n_div+2*n_frames;
    SpMat HsF = SpMat(num_vars,num_vars);
    GetSparseAnaJacobian(ptx, pty, pptx, ppty, o, op, id, idp, w, h, div, n_frames, n, a, b, J);
    Jt = J.transpose();
    std::cout << "Taken Transpose\n";
    HsF = Jt*J;
    B = Jt*R;
    Eigen::MatrixXd H =  Eigen::MatrixXd(HsF);
    nz_rows.clear();
    for(int i=0; i<num_vars; i++)
    {
        bool foundnz = false;
        for(int j=0; j<num_vars; j++)
        {
            if(H(i,j) != 0)
                foundnz = true;
        }
        if(foundnz) nz_rows.push_back(i);        
    }
    Eigen::MatrixXd rdH(nz_rows.size(), nz_rows.size());
    for(int i=0; i<nz_rows.size(); i++)
    {
        for(int j=0; j<nz_rows.size(); j++)
        {
            rdH(i,j) = H(nz_rows[i], nz_rows[j]);
            if(i==j)
            {
                rdH(i,j) = rdH(i,j) + lambda*H(nz_rows[i], nz_rows[j]);
            }
        }
    }
    Eigen::MatrixXd rdB(nz_rows.size(),1);
    for(int i=0; i<nz_rows.size(); i++)
    {
        rdB(i,0) = B(nz_rows[i],0);
    }
    B = rdB;
    Hs = SpMat(nz_rows.size(),nz_rows.size());
    Hs = rdH.sparseView();
    //std::cout << "\n" << rdH.block(0,0,50,50) << "\n";
    
    std::cout << (double)rdB.size()/n_div << " Linearize System End\n";
}

void IRPhotoCalib::GetResiduals(vector<int> &ptx, vector<int>& pty, 
                            vector<int>& pptx, vector<int>& ppty, 
                            vector<float>& o, vector<float>& op, 
                            vector<int>& id, vector<int>& idp, 
                            int w, int h, int div, int n_frames,
                      vector<double> n, vector<double> a, vector<double> b,
                      Eigen::MatrixXd &R)
{
    int num_meas = o.size();
    R = Eigen::MatrixXd(num_meas, 1);
    std::cout << "Get Residual Start\n";
    for(int i=0; i<o.size(); i++)
    {
        double ai = a[id[i]];
        double bi = b[id[i]];
        double aip = a[idp[i]];
        double bip = b[idp[i]];
        
        double nxy = n[getNid(ptx[i], pty[i], div, w)];
        double nxyp = n[getNid(pptx[i], ppty[i], div, w)];
        
        // double t = o[i]*exp(ai)+bi+nxy;
        // double tp = op[i]*exp(aip)+bip+nxyp;
        // R(i,0) = t-tp;
        R(i,0) = o[i] - ( op[i]*(aip-bip) + bip - nxyp - bi + nxy ) / (ai-bi);
    }
    // for(int i=0; i<n_frames; i++)
    // {
    //     R(num_meas+i,0) = a[i];
    //     R(num_meas+i+n_frames,0) = b[i];
    // }
    std::cout << "Get Residual End\n";
}

void IRPhotoCalib::GetSparseAnaJacobian(vector<int> &ptx, vector<int>& pty, 
                            vector<int>& pptx, vector<int>& ppty, 
                            vector<float>& o, vector<float>& op, 
                            vector<int>& id, vector<int>& idp, 
                            int w, int h, int div, int n_frames,
                      vector<double> n, vector<double> a, vector<double> b,
                      SpMat &J)
{
    int num_meas = o.size();
    int n_div = w*h/(div*div);
    int num_vars = n_div+2*n_frames;
    std::vector<bool> nzr(num_meas);
    for (int i=0; i<nzr.size(); i++) nzr[i]=false;
    J = SpMat(num_meas, num_vars);
    typedef Eigen::Triplet<double> T;
    std::vector<T> tripletList;
    //tripletList.reserve(num_meas+2*n_frames);
    std::cout << "Get Jacobian Start\n";
    // Setup Jacobian Convetion: 2120 + a + b. And t-tp
    int total_n = n_div;
    for(int i=0; i<o.size(); i++)
    {
        int f_num = id[i]; int fp_num = idp[i];
        int n_num = getNid(ptx[i], pty[i], div, w);
        int np_num = getNid(pptx[i], ppty[i], div, w);

        double ai = a[f_num];
        double aip = a[fp_num];
        double bi = b[f_num];
        double bip = b[fp_num];
        double nxy = n[n_num];
        double nxyp = n[np_num];
        
        int ai_id = total_n+f_num;
        int aip_id = total_n+fp_num;
        int bi_id = total_n+n_frames+f_num;
        int bip_id = total_n+n_frames+fp_num;
        int nxy_id = n_num;
        int nxyp_id = np_num;
        tripletList.push_back(T(i,ai_id,(double)    (op[i]*aip-op[i]*bip+bip-nxyp-bi-nxy)/( (ai-bi)*(ai-bi) )          ));
        tripletList.push_back(T(i,aip_id,(double)  -op[i]/(ai-bi)  ));
        tripletList.push_back(T(i,bi_id,(double)  -(op[i]*aip-op[i]*bip+bip-nxyp-bi-nxy)/( (ai-bi)*(ai-bi) ) ));
        tripletList.push_back(T(i,bip_id,(double)  (op[i]-1.0)/(ai-bi)   ));
        tripletList.push_back(T(i,nxy_id,(double)  -1.0/(ai-bi)    ));
        tripletList.push_back(T(i,nxyp_id,(double)  1.0/(ai-bi)   ));
        nzr[i] = true;
    }
    for(int i=0; i<nzr.size(); i++){if(!nzr[i])std::cout<<"BIGGGG ISSUE\n";}
    J.setFromTriplets(tripletList.begin(), tripletList.end());
    std::cout << "Get Jacobian End\n";
}

void IRPhotoCalib::RunGNAOptimizerAB(vector<int>& ptx, vector<int>& pty, 
                            vector<int> &pptx, vector<int> &ppty, 
                            vector<float> &o, vector<float> &op, 
                            vector<int> &id, vector<int> &idp, 
                            int &w, int& h, int &div, int& n_frames,
                            vector<double> &out_s, vector<double> &n,
                            vector<double> &out_a, vector<double> &out_b)
{
    std::cout << "Starting Optimizer\n";
    vector<double> a(n_frames); vector<double> b(n_frames);
    for (int i=0; i<n_frames; i++)
    { a[i] = 0; b[i] = 0; }
    
    Eigen::VectorXd dX;
    for (int i=0; i<1; i++)
    {
        std::cout << "Optimizer Loop Start\n";
        double t_op_start = get_wall_time();
        //Linearize the system
        Eigen::MatrixXd B;
        SpMat Hs;
        LinearizeSystemAB(ptx, pty, pptx, ppty, o, op, id, idp, w, h, div, n_frames, n, a, b, B, Hs);
        
        //COLAMD ordering
        
        Hs.makeCompressed();
        Eigen::SimplicialLLT <SpMat > solver;
        solver.compute(Hs);
        if(solver.info()!=Eigen::Success) {
        // decomposition failed
            cout << "failed\n";
        }
        dX = solver.solve(B);
        if(solver.info()!=Eigen::Success) {
            cout << "failed\n";
        }
            
        double t_op_end = get_wall_time();
        std::cout << t_op_end-t_op_start << " solved in time\n";
        
        //make changes
        for(int j=0; j<a.size(); j++)
            a[j] = a[j]-dX(j,0);
        for(int j=0; j<b.size(); j++)
            b[j] = b[j]-dX(j+n_frames,0);
        //if(solver.error() < 1.0e-3) break;
    }
    
    out_a = a; out_b = b;
    for (int i=0; i<n_frames; i++)
    { out_a[i] = out_a[i]; out_b[i] = out_b[i]; }
}

void IRPhotoCalib::LinearizeSystemAB(vector<int> &ptx, vector<int>& pty, 
                            vector<int>& pptx, vector<int>& ppty, 
                            vector<float>& o, vector<float>& op, 
                            vector<int>& id, vector<int>& idp, 
                            int &w, int &h, int &div, int &n_frames,
                      vector<double> &n, vector<double> &a, vector<double> &b,
                      Eigen::MatrixXd &B, SpMat &Hs)
{
    std::cout << "Linearize SystemAB Start\n";
    Eigen::MatrixXd R;
    GetResidualsAB(ptx, pty, pptx, ppty, o, op, id, idp, w, h, div, n_frames, n, a, b, R);
    
    SpMat J, Jt;
    int num_vars = 2*n_frames;
    Hs = SpMat(num_vars,num_vars);
    GetSparseAnaJacobianAB(ptx, pty, pptx, ppty, o, op, id, idp, w, h, div, n_frames, n, a, b, J);
    Jt = J.transpose();
    std::cout << "Taken Transpose\n";
    Hs = Jt*J;
    B = Jt*R;
    
    Eigen::MatrixXd H =  Eigen::MatrixXd(Hs);
    vector<int> nz_rows;
    for(int i=0; i<num_vars; i++)
    {
        bool foundnz = false;
        for(int j=0; j<num_vars; j++)
        {
            if(H(i,j) != 0)
                foundnz = true;
            // if(i==j)
            // {
            //     H(i,j) = H(i,j) + 1.5*H(i,j);
            // }
        }
        if(foundnz) nz_rows.push_back(i);        
    }
    //Hs = H.sparseView();
    if(nz_rows.size() < 2*n_frames) std::cout << "Big Error!!!!!!!!!!!!!!!\n";
    std::cout << "Linearize SystemAB End\n";
}

void IRPhotoCalib::GetResidualsAB(vector<int> &ptx, vector<int>& pty, 
                            vector<int>& pptx, vector<int>& ppty, 
                            vector<float>& o, vector<float>& op, 
                            vector<int>& id, vector<int>& idp, 
                            int &w, int &h, int& div, int &n_frames,
                      vector<double>& n, vector<double> &a, vector<double> &b,
                      Eigen::MatrixXd &R)
{
    int num_meas = o.size();
    R = Eigen::MatrixXd(num_meas+2*n_frames, 1);
    std::cout << "Get ResidualAB Start\n";
    for(int i=0; i<o.size(); i++)
    {
        double ai = a[id[i]];
        double bi = b[id[i]];
        double aip = a[idp[i]];
        double bip = b[idp[i]];
        
        double nxy = n[getNid(ptx[i], pty[i], div, w)];
        double nxyp = n[getNid(pptx[i], ppty[i], div, w)];
        
        double t = o[i]*exp(ai)+bi+nxy;
        double tp = op[i]*exp(aip)+bip+nxyp;
        R(i,0) = t-tp;
    }
    for(int i=0; i<n_frames; i++)
    {
        R(num_meas+i,0) = (a[i]);
        R(num_meas+i+n_frames,0) = (b[i]);
    }
    std::cout << "Get ResidualAB End\n";
}

void IRPhotoCalib::GetSparseAnaJacobianAB(vector<int> &ptx, vector<int>& pty, 
                            vector<int>& pptx, vector<int>& ppty, 
                            vector<float>& o, vector<float>& op, 
                            vector<int>& id, vector<int>& idp, 
                            int &w, int &h, int &div, int& n_frames,
                      vector<double> &n, vector<double> &a, vector<double> &b,
                      SpMat &J)
{
    int num_meas = o.size();
    int num_vars = 2*n_frames;
    J = SpMat(num_meas+2*n_frames, num_vars);
    typedef Eigen::Triplet<double> T;
    std::vector<T> tripletList;
    tripletList.reserve(num_meas+2*n_frames);
    std::cout << "Get JacobianAB Start\n";
    // Setup Jacobian Convetion: a + b. And t-tp
    for(int i=0; i<o.size(); i++)
    {
        double ai = a[id[i]];
        double aip = a[idp[i]];
        int f_num = id[i]; int fp_num = idp[i];
        
        int ai_id = f_num;
        int aip_id = fp_num;
        int bi_id = n_frames+f_num;
        int bip_id = n_frames+fp_num;
        tripletList.push_back(T(i,ai_id,(double)o[i]*exp(ai)));
        tripletList.push_back(T(i,aip_id,-(double)op[i]*exp(aip)));
        tripletList.push_back(T(i,bi_id,1.0));
        tripletList.push_back(T(i,bip_id,-1.0));
    }
    for(int i=0; i<n_frames; i++)
    {
        int ai_id = i;
        int bi_id = n_frames+i;
        tripletList.push_back(T(num_meas+i, ai_id, 1.0));
        tripletList.push_back(T(num_meas+i+n_frames, bi_id, 1.0));
    }
    J.setFromTriplets(tripletList.begin(), tripletList.end());
    std::cout << "Get JacobianAB End\n";
}

void IRPhotoCalib::GetAnaJacobian(vector<int> &ptx, vector<int>& pty, 
                            vector<int>& pptx, vector<int>& ppty, 
                            vector<float>& o, vector<float>& op, 
                            vector<int>& id, vector<int>& idp, 
                            int w, int h, int div, int n_frames,
                      vector<double> n, vector<double> a, vector<double> b,
                      Eigen::MatrixXd &J)
{
    std::cout << "Get Jacobian Start\n";
    int num_meas = o.size();
    int n_div = w*h/(div*div);
    int num_vars = n_div+2*n_frames;
    J = Eigen::MatrixXd::Zero(num_meas, num_vars);
    // Setup Jacobian Convetion: 2120 + a + b. And t-tp
    int total_n = n_div;
    for(int i=0; i<o.size(); i++)
    {
        double ai = a[id[i]];
        double aip = a[idp[i]];
        int f_num = id[i]; int fp_num = idp[i];
        int n_num = getNid(ptx[i], pty[i], div, w);
        int np_num = getNid(pptx[i], ppty[i], div, w);
        
        int ai_id = total_n+f_num;
        int aip_id = total_n+fp_num;
        int bi_id = total_n+n_frames+f_num;
        int bip_id = total_n+n_frames+fp_num;
        int nxy_id = n_num;
        int nxyp_id = np_num;
        J(i, ai_id) = o[i]*exp(ai);        
        J(i, aip_id) = -op[i]*exp(aip);
        J(i, bi_id) = 1.0;
        J(i, bip_id) = -1.0;
        J(i, nxy_id) = 1.0;
        J(i, nxyp_id) = 1.0;
    }
    std::cout << "Get Jacobian End\n";
}


void IRPhotoCalib::RunNewOptimizer(vector<int> ptx, vector<int> pty, 
                            vector<int> pptx, vector<int> ppty, 
                            vector<float> o, vector<float> op, 
                            vector<int> id, vector<int> idp, 
                            int w, int h, int div, int n_frames,
                            vector<double> &out_s, vector<double> &out_n,
                            vector<double> &out_a, vector<double> &out_b)
{
  int n_pairs = ptx.size();
  vector<double> s(w*h/(div*div)); vector<double> n(w*h/(div*div)); vector<double> a(n_frames); vector<double> b(n_frames);
  for (int i=0; i<w*h/(div*div); i++)
  { s[i] = 0; n[i] = 0.000001; }
  for (int i=0; i<n_frames; i++)
  { a[i] = 0; b[i] = 0; }
  cout << "Making the problen\n";
  ceres::Problem problem;
  for (int i=0; i<n_pairs; i++)
  {
    problem.AddResidualBlock(
      new ceres::AutoDiffCostFunction<NewPhotometricError, 1, 2120, 30, 30>(
              new NewPhotometricError(o[i], op[i], id[i], idp[i], (int)(std::floor(pty[i]/div) * std::floor(w/div) + std::floor(ptx[i]/div)), (int)(std::floor(ppty[i]/div) * std::floor(w/div) + std::floor(pptx[i]/div)) )
      ), NULL,
      &n[0],&a[0],&b[0]
    );
  }

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
  options.minimizer_progress_to_stdout = true;
  options.max_num_iterations = 6;
  ceres::Solver::Summary summary;
  cout << "Calling the solver\n";
  ceres::Solve(options, &problem, &summary);
  
  out_a = a; out_b = b; out_n = n;
  //out_a.push_back(0.0); out_b.push_back(0.0);
}

void IRPhotoCalib::RunNewGainOptimizer(vector<int> ptx, vector<int> pty, 
                            vector<int> pptx, vector<int> ppty, 
                            vector<float> o, vector<float> op, 
                            vector<int> id, vector<int> idp, 
                            int w, int h, int div, int n_frames,
                            vector<double> &n,
                            vector<double> &out_a, vector<double> &out_b)
{
  int n_pairs = ptx.size();
  vector<double> a(n_frames); vector<double> b(n_frames);
  for (int i=0; i<n_frames; i++)
  { a[i] = 0; b[i] = 0; }
  cout << "Making the problen\n";
  ceres::Problem problem;
  for (int i=0; i<n_pairs; i++)
  {
    problem.AddResidualBlock(
      new ceres::AutoDiffCostFunction<NewGainError, 1, 30, 30>(
              new NewGainError(o[i], op[i], id[i], idp[i], n[(int)(std::floor(pty[i]/div) * std::floor(w/div) + std::floor(ptx[i]/div))], n[(int)(std::floor(ppty[i]/div) * std::floor(w/div) + std::floor(pptx[i]/div))] )
      ), NULL,
      &a[0],&b[0]
    );
  }
  

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  options.minimizer_progress_to_stdout = true;
  //options.max_num_iterations = 16;
  ceres::Solver::Summary summary;
  cout << "Calling the solver\n";
  ceres::Solve(options, &problem, &summary);
  
  out_a = a; out_b = b;
  //out_a.push_back(0.0); out_b.push_back(0.0);
}

bool IRPhotoCalib::isInsideFrame(int x, int y, int w, int h)
{
  if(x>=0 && x<w && y>=0 && y<h)
    return true;
  return false;
}

float IRPhotoCalib::getKernelO(cv::Mat& im, int x, int y, int k)
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

void IRPhotoCalib::getStartPoints(vector< int > ptx, vector< int > pty, vector< int > pptx, vector< int > ppty, int w, int h, int div)
{
  int n_div = w*h/(div*div);
  for(int i=0; i<ptx.size(); i++)
  {
    int nid = getNid(ptx[i], pty[i], div, w);
    int nidp = getNid(pptx[i], ppty[i], div, w);
  }
}

int IRPhotoCalib::SerializeLocation(int x, int y, int w, int h)
{return y*w+x;}

void IRPhotoCalib::DeSerializeLocation(int id, int w, int h, int & x, int & y)
{
  y = (int)(id / w);
  x = (int)(id % w);
}

void IRPhotoCalib::getRelativeGains(double a1, double b1, double a2, double b2, double & a12, double & b12){
  double e12 = (a2-b2)/(a1-b1);
  b12 = (b2-b1)/(a1-b1);
  a12 = e12 + b12;
}

void IRPhotoCalib::EstimateSpatialParameters(vector<int> &ptx, vector<int>& pty, 
                                   vector<int>& pptx, vector<int>& ppty, 
                                   vector<float>& o, vector<float>& op, 
                                   vector<int>& id, vector<int>& idp,
                                   vector<double> & all_aips, vector<double> & all_bips,
                                   int w, int h, int div,
                                   string PS_correspondance, string PS_location, string A_location)
{
  vector<int> fptx, fpty, fpptx, fppty, fid, fidp;  
  vector<float> fo, fop;

  vector<int> serial_to_variable_id((w*h)/(div*div),-1);
  vector<int> variable_to_serial_id;
  vector< pair<int, int> > Aposneg_id;
  vector<double> vecB;
  vector<int> correscount((w*h)/(div*div),0);
  int max_correscount = 10;

  for (int i=0; i<ptx.size(); i++)
  {
    //if( abs(ptx[i]-pptx[i]) < div && abs(pty[i]-ppty[i]) < div ) continue;

    int sidp = getNid(pptx[i], ppty[i], div, w);
    int sid = getNid(ptx[i], pty[i], div, w);
    
    if (sidp == sid) continue;
    if(correscount[sidp] > max_correscount && correscount[sid] > max_correscount) continue;

    double tat1, tbt1;
    getRelativeGains(all_aips[id[i]+1], all_bips[id[i]+1], all_aips[idp[i]+1], all_bips[idp[i]+1], tat1, tbt1);
    //cout << tat1 << ' ' << tbt1 << '\n';
    double bi = op[i]*(tat1-tbt1) - o[i] + tbt1;
    vecB.push_back(bi);
    // int sidp = SerializeLocation(pptx[i], ppty[i], w, h);
    // int sid = SerializeLocation(ptx[i], pty[i], w, h);
    
    int vidp = -1; int vid = -1;
    if (serial_to_variable_id[sidp]==-1)
    {
      // This variable is first. Thus add to serial_to_variable_id and variable_to_serial_id
      variable_to_serial_id.push_back(sidp);
      int thisVid = variable_to_serial_id.size()-1;
      serial_to_variable_id[sidp] = thisVid;
      vidp = thisVid;
    }
    else vidp = serial_to_variable_id[sidp];

    if (serial_to_variable_id[sid]==-1)
    {
      // This variable is first. Thus add to serial_to_variable_id and variable_to_serial_id
      variable_to_serial_id.push_back(sid);
      int thisVid = variable_to_serial_id.size()-1;
      serial_to_variable_id[sid]= thisVid;
      vid = thisVid;
    }
    else vid = serial_to_variable_id[sid];

    Aposneg_id.push_back(make_pair(vidp,vid));
    correscount[sid]++; correscount[sidp]++;  
    fptx.push_back(ptx[i]); fpptx.push_back(pptx[i]); 
    fpty.push_back(pty[i]); fppty.push_back(ppty[i]);
    fo.push_back(o[i]); fop.push_back(op[i]);
    fid.push_back(id[i]); fidp.push_back(idp[i]);
  }

  exportCorrespondences(fptx, fpty, fpptx, fppty, fo, fop, fid, fidp, PS_correspondance);

  cout << "Size of A matrix " << vecB.size() << " " << variable_to_serial_id.size() << endl;
  
  SpMat A(vecB.size(), variable_to_serial_id.size());
  Eigen::MatrixXd b(vecB.size(),1);
  std::vector<T> tripletList; tripletList.reserve(vecB.size()*2);
  for (int i=0; i<vecB.size(); i++)
  {
    b(i,0) = vecB[i];
    tripletList.push_back(T(i,Aposneg_id[i].first,1.0));
    tripletList.push_back(T(i,Aposneg_id[i].second,-1.0));
  }
  A.setFromTriplets(tripletList.begin(), tripletList.end());
  
  Eigen::SparseQR <SpMat, Eigen::COLAMDOrdering<int> > solver;
  //Eigen::LeastSquaresConjugateGradient < SpMat > solver;

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

  ofstream result_file, resultA_file;
  result_file.open(PS_location);
  resultA_file.open(A_location);
  for (int i=0; i<variable_to_serial_id.size(); i++){
    result_file << variable_to_serial_id[i] << ',' << x(i) << '\n';
  }
  result_file.close();

  for (int i=0; i<Aposneg_id.size(); i++){
    resultA_file << Aposneg_id[i].first << ',' << Aposneg_id[i].second << '\n';
  }
  resultA_file.close();
}

void IRPhotoCalib::exportCorrespondences(vector<int> &ptx, vector<int>& pty, 
                           vector<int>& pptx, vector<int>& ppty, 
                           vector<float>& o, vector<float>& op, 
                           vector<int>& id, vector<int>& idp, string filename)
{
  ofstream c_file;
  c_file.open(filename);
  for (int i=0; i<ptx.size(); i++){
    c_file <<ptx[i]<<','<<pptx[i]<<','<<pty[i]<<','<<ppty[i]<<','<<o[i]<<','<<op[i]<<','<<id[i]<<','<<idp[i]<<'\n';
  }
  c_file.close();
}

void IRPhotoCalib::exportKFPT(vector<double> & all_aips, vector<double> & all_bips, string filename)
{
  ofstream c_file;
  c_file.open(filename);
  for (int i=0; i<all_aips.size(); i++){
    c_file <<i<<','<<all_aips[i]<<','<<all_bips[i]<<'\n';
  }
  c_file.close();
}