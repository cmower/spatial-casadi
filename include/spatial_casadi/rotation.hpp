#pragma once
#include <spatial_casadi/detail.hpp>
#include <random>

namespace spatial_casadi {
/// Rotations stored as scalar-last quaternion columns (4 x N).
template<class T> class Rotation {
 public:
  explicit Rotation(const T& quaternion, bool normalize = true, bool scalar_first = false) {
    detail::shape(quaternion, 4);
    q_ = quaternion;
    if (scalar_first)
      q_ = T::vertcat({T(q_(casadi::Slice(1,4), casadi::Slice())), T(q_(0,casadi::Slice()))});
    if (normalize) q_ = detail::columns(q_, [](const T& q) { return detail::normalize(q); });
  }
  static Rotation from_quat(const T& q, bool scalar_first = false) {
    return Rotation(q, true, scalar_first);
  }
  static Rotation identity(casadi_int num = 1) {
    detail::require(num > 0, "Batch must be nonempty.");
    T q = T::zeros(4,num); q(3,casadi::Slice()) = 1;
    return Rotation(q, false);
  }
  static Rotation concatenate(const std::vector<Rotation>& rotations) {
    detail::require(!rotations.empty(), "Provide at least one rotation.");
    std::vector<T> q;
    for (const auto& r : rotations) q.push_back(r.as_quat());
    return Rotation(T::horzcat(q),false);
  }
  static Rotation random(casadi_int num, unsigned long seed) {
    detail::require(num > 0, "Batch must be nonempty.");
    std::mt19937 generator(seed);
    std::normal_distribution<double> normal;
    casadi::DM q=casadi::DM::zeros(4,num);
    for (casadi_int i=0;i<q.numel();++i) q(i)=normal(generator);
    return Rotation(T(q));
  }
  static Rotation from_rotvec(const T& v, bool degrees = false) {
    detail::shape(v,3);
    T radians = degrees ? v*(detail::pi/180) : v;
    return Rotation(detail::columns(radians, [](const T& x) { return detail::exp_quat(x); }), false);
  }
  static Rotation from_mrp(const T& p) {
    detail::shape(p,3);
    return Rotation(detail::columns(p, [](const T& x) {
      T s = dot(x,x);
      return detail::cat<T>({2*x/(1+s), (1-s)/(1+s)});
    }), false);
  }
  static Rotation from_matrix(const T& blocks) {
    detail::require(blocks.size1() == 3 && blocks.size2() > 0 && blocks.size2()%3 == 0,
                    "Rotation matrices must have shape 3 x (3*N).");
    std::vector<T> result;
    for (casadi_int i = 0; i < blocks.size2(); i += 3) {
      T m = blocks(casadi::Slice(),casadi::Slice(i,i+3));
      if constexpr (std::is_same_v<T,casadi::DM>) {
        detail::require(m.is_regular(), "Matrix must be finite.");
        detail::require(static_cast<double>(norm_fro(mtimes(m.T(),m)-T::eye(3))) < 1e-6 &&
                        static_cast<double>(det(m)) > 0, "Matrix must be a proper rotation.");
      }
      result.push_back(detail::matrix_quat(m));
    }
    return Rotation(T::horzcat(result), false);
  }
  static Rotation from_euler(const std::string& seq, const T& angles, bool degrees = false) {
    bool intrinsic = detail::euler_sequence(seq);
    detail::shape(angles, seq.size());
    T a = degrees ? angles*(detail::pi/180) : angles;
    std::vector<T> result;
    for (casadi_int n = 0; n < a.size2(); ++n) {
      T q = detail::cat<T>({T(0),T(0),T(0),T(1)});
      for (size_t i = 0; i < seq.size(); ++i) {
        T e = T::zeros(4,1); e(3)=cos(a(i,n)/2); e(detail::axis(seq[i]))=sin(a(i,n)/2);
        q = intrinsic ? detail::multiply(q,e) : detail::multiply(e,q);
      }
      result.push_back(q);
    }
    return Rotation(T::horzcat(result), false);
  }
  T as_quat(bool canonical = false, bool scalar_first = false) const {
    T q = canonical ? detail::columns(q_, [](const T& x) { return detail::canonical(x); }) : q_;
    return scalar_first ? T::vertcat({T(q(3,casadi::Slice())),T(q(casadi::Slice(0,3),casadi::Slice()))}) : q;
  }
  T as_matrix() const {
    return detail::columns(q_, [](const T& q) {
      T x=q(0), y=q(1), z=q(2), w=q(3);
      return T::horzcat({detail::cat<T>({1-2*(y*y+z*z),2*(x*y+z*w),2*(x*z-y*w)}),
                        detail::cat<T>({2*(x*y-z*w),1-2*(x*x+z*z),2*(y*z+x*w)}),
                        detail::cat<T>({2*(x*z+y*w),2*(y*z-x*w),1-2*(x*x+y*y)})});
    });
  }
  T as_rotvec(bool degrees = false) const {
    T v = detail::columns(q_, [](const T& x) { return detail::log_quat(x); });
    return degrees ? v*(180/detail::pi) : v;
  }
  T as_mrp() const {
    return detail::columns(q_, [](const T& x) {
      T q = detail::canonical(x);
      return T(q(casadi::Slice(0,3)))/(1+q(3));
    });
  }
  T as_euler(const std::string& sequence, bool degrees = false) const {
    bool intrinsic = detail::euler_sequence(sequence,true);
    std::string seq = sequence;
    if (intrinsic) std::reverse(seq.begin(),seq.end());
    int i=detail::axis(seq[0]), j=detail::axis(seq[1]), k=detail::axis(seq[2]);
    bool symmetric = i==k;
    if (symmetric) k=3-i-j;
    int sign=(i-j)*(j-k)*(k-i)/2;
    T result = detail::columns(q_, [&](const T& q) {
      T a,b,c,d;
      if (symmetric) { a=q(3); b=q(i); c=q(j); d=q(k)*sign; }
      else { a=q(3)-q(j); b=q(i)+q(k)*sign; c=q(j)+q(3); d=q(k)*sign-q(i); }
      T middle=2*atan2(sqrt(c*c+d*d),sqrt(a*a+b*b));
      T low=fabs(middle)<=1e-7, high=fabs(middle-detail::pi)<=1e-7;
      T sum=atan2(b,a), diff=atan2(d,c);
      T first=T::if_else(low,2*sum,T::if_else(high,2*diff*(intrinsic ? 1 : -1),
                                           intrinsic ? sum+diff : sum-diff));
      T third=T::if_else(low+high, T(0), intrinsic ? sum-diff : sum+diff);
      if (!symmetric) { if (intrinsic) first*=sign; else third*=sign; middle-=detail::pi/2; }
      T angles=detail::cat<T>({first,middle,third});
      for (int n=0;n<3;++n) angles(n)=T::if_else(angles(n)<-detail::pi,angles(n)+2*detail::pi,
                                        T::if_else(angles(n)>detail::pi,angles(n)-2*detail::pi,T(angles(n))));
      return angles;
    });
    return degrees ? result*(180/detail::pi) : result;
  }
  Rotation compose(const Rotation& other) const {
    casadi_int n=detail::count(size(),other.size());
    std::vector<T> result;
    for (casadi_int i=0;i<n;++i) result.push_back(detail::multiply(detail::col(q_,i),detail::col(other.q_,i)));
    return Rotation(T::horzcat(result),false);
  }
  Rotation operator*(const Rotation& other) const { return compose(other); }
  Rotation inv() const {
    return Rotation(detail::columns(q_, [](const T& q) { return detail::conjugate(q); }),false);
  }
  Rotation power(double n) const {
    detail::require(std::isfinite(n), "Power must be finite.");
    if (n==0) return identity(size());
    if (n==1) return *this;
    if (n==-1) return inv();
    return from_rotvec(as_rotvec()*n);
  }
  T apply(const T& vectors, bool inverse = false) const {
    detail::shape(vectors,3);
    casadi_int n=detail::count(size(),vectors.size2());
    std::vector<T> result;
    for (casadi_int i=0;i<n;++i) {
      T q=detail::col(q_,i);
      if (inverse) q=detail::conjugate(q);
      result.push_back(detail::rotate(q,detail::col(vectors,i)));
    }
    return T::horzcat(result);
  }
  T magnitude() const {
    return detail::columns(q_, [](const T& q) {
      return 2*atan2(norm_2(T(q(casadi::Slice(0,3)))),fabs(q(3)));
    });
  }
  T approx_equal(const Rotation& other, double atol = 1e-8, bool degrees = false) const {
    detail::require(atol >= 0, "Tolerance must be nonnegative.");
    return inv().compose(other).magnitude() < (degrees ? atol*detail::pi/180 : atol);
  }
  casadi_int size() const { return q_.size2(); }
  Rotation at(casadi_int index) const {
    detail::require(index >= 0 && index < size(), "Rotation index out of range.");
    return Rotation(T(q_(casadi::Slice(),index)),false);
  }
 private:
  T q_;
};
}
