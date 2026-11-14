#pragma once
#include <spatial_casadi/rotation.hpp>

namespace spatial_casadi {
/// Rigid transforms: p_parent = rotation.apply(p_child) + translation.
template<class T> class RigidTransform {
 public:
  RigidTransform(const T& translation, const Rotation<T>& rotation) : r_(rotation), t_(translation) {
    detail::shape(t_,3);
    casadi_int n=detail::count(r_.size(),t_.size2());
    if (r_.size()==1 && n>1) r_=Rotation<T>(repmat(r_.as_quat(),1,n),false);
    if (t_.size2()==1 && n>1) t_=repmat(t_,1,n);
  }
  static RigidTransform from_components(const T& translation, const Rotation<T>& rotation) {
    return RigidTransform(translation,rotation);
  }
  static RigidTransform identity(casadi_int num = 1) {
    return RigidTransform(T::zeros(3,num),Rotation<T>::identity(num));
  }
  static RigidTransform concatenate(const std::vector<RigidTransform>& transforms) {
    detail::require(!transforms.empty(), "Provide at least one transform.");
    std::vector<Rotation<T>> rotations;
    std::vector<T> translations;
    for (const auto& tf : transforms) {
      rotations.push_back(tf.rotation()); translations.push_back(tf.translation());
    }
    return RigidTransform(T::horzcat(translations),Rotation<T>::concatenate(rotations));
  }
  static RigidTransform from_translation(const T& translation) {
    return RigidTransform(translation,Rotation<T>::identity());
  }
  static RigidTransform from_rotation(const Rotation<T>& rotation) {
    return RigidTransform(T::zeros(3,1),rotation);
  }
  static RigidTransform from_matrix(const T& blocks) {
    detail::require(blocks.size1()==4 && blocks.size2()>0 && blocks.size2()%4==0,
                    "Transform matrices must have shape 4 x (4*N).");
    std::vector<T> rotations,translations;
    for (casadi_int i=0;i<blocks.size2();i+=4) {
      if constexpr (std::is_same_v<T,casadi::DM>) {
        T bottom=blocks(3,casadi::Slice(i,i+4));
        detail::require(bottom.is_regular() &&
          static_cast<double>(norm_fro(bottom-T::horzcat({T(0),T(0),T(0),T(1)}))) < 1e-12,
          "Last matrix row must be [0, 0, 0, 1].");
      }
      rotations.push_back(blocks(casadi::Slice(0,3),casadi::Slice(i,i+3)));
      translations.push_back(blocks(casadi::Slice(0,3),i+3));
    }
    return RigidTransform(T::horzcat(translations),Rotation<T>::from_matrix(T::horzcat(rotations)));
  }
  static RigidTransform from_exp_coords(const T& coordinates) {
    detail::shape(coordinates,6);
    T translation=detail::columns(coordinates,[](const T& x) {
      T w=x(casadi::Slice(0,3)), v=x(casadi::Slice(3,6));
      T s=dot(w,w), small=s<1e-2, safe=T::if_else(small,T(1),s), a=sqrt(safe);
      T A=T::if_else(small,0.5-s/24+s*s/720-s*s*s/40320+s*s*s*s/3628800,(1-cos(a))/safe);
      T B=T::if_else(small,1.0/6-s/120+s*s/5040-s*s*s/362880+s*s*s*s/39916800,(a-sin(a))/(safe*a));
      T wv=cross(w,v);
      return v+A*wv+B*cross(w,wv);
    });
    return RigidTransform(translation,Rotation<T>::from_rotvec(T(coordinates(casadi::Slice(0,3),casadi::Slice()))));
  }
  T as_exp_coords() const {
    T w=r_.as_rotvec();
    std::vector<T> result;
    for (casadi_int i=0;i<size();++i) {
      T v=detail::col(w,i), t=detail::col(t_,i), s=dot(v,v), small=s<1e-2;
      T safe=T::if_else(small,T(1),s), a=sqrt(safe);
      T C=T::if_else(small,1.0/12+s/720+s*s/30240+s*s*s/1209600+s*s*s*s/47900160,
                    (1-a*cos(a/2)/(2*sin(a/2)))/safe);
      T vt=cross(v,t);
      result.push_back(detail::cat<T>({v,t-0.5*vt+C*cross(v,vt)}));
    }
    return T::horzcat(result);
  }
  static RigidTransform from_dual_quat(const T& dual, bool scalar_first = false) {
    detail::shape(dual,8);
    std::vector<T> rotations, translations;
    for (casadi_int i=0;i<dual.size2();++i) {
      T x=detail::col(dual,i), real=x(casadi::Slice(0,4)), imag=x(casadi::Slice(4,8));
      if (scalar_first) {
        real=detail::cat<T>({T(real(casadi::Slice(1,4))),T(real(0))});
        imag=detail::cat<T>({T(imag(casadi::Slice(1,4))),T(imag(0))});
      }
      T q=detail::normalize(real);
      imag/=norm_2(real);
      imag-=q*dot(q,imag);
      T t=2*detail::multiply(imag,detail::conjugate(q));
      rotations.push_back(q); translations.push_back(t(casadi::Slice(0,3)));
    }
    return RigidTransform(T::horzcat(translations),Rotation<T>(T::horzcat(rotations),false));
  }
  T as_dual_quat(bool scalar_first = false) const {
    std::vector<T> result;
    for (casadi_int i=0;i<size();++i) {
      T real=detail::col(r_.as_quat(),i);
      T imag=0.5*detail::multiply(detail::cat<T>({detail::col(t_,i),T(0)}),real);
      if (scalar_first) {
        real=detail::cat<T>({T(real(3)),T(real(casadi::Slice(0,3)))});
        imag=detail::cat<T>({T(imag(3)),T(imag(casadi::Slice(0,3)))});
      }
      result.push_back(detail::cat<T>({real,imag}));
    }
    return T::horzcat(result);
  }
  T as_matrix() const {
    T matrices=r_.as_matrix();
    std::vector<T> result;
    for (casadi_int i=0;i<size();++i) {
      T top=T::horzcat({T(matrices(casadi::Slice(),casadi::Slice(3*i,3*i+3))),detail::col(t_,i)});
      result.push_back(T::vertcat({top,T::horzcat({T(0),T(0),T(0),T(1)})}));
    }
    return T::horzcat(result);
  }
  RigidTransform compose(const RigidTransform& other) const {
    return RigidTransform(r_.apply(other.t_)+t_,r_.compose(other.r_));
  }
  RigidTransform operator*(const RigidTransform& other) const { return compose(other); }
  RigidTransform inv() const { return RigidTransform(-r_.apply(t_,true),r_.inv()); }
  RigidTransform power(double n) const {
    detail::require(std::isfinite(n), "Power must be finite.");
    if (n==0) return identity(size());
    if (n==1) return *this;
    if (n==-1) return inv();
    return from_exp_coords(as_exp_coords()*n);
  }
  T apply(const T& vectors, bool inverse = false) const {
    detail::shape(vectors,3);
    detail::count(size(),vectors.size2());
    return inverse ? r_.apply(vectors-t_,true) : r_.apply(vectors)+t_;
  }
  Rotation<T> rotation() const { return r_; }
  T translation() const { return t_; }
  casadi_int size() const { return r_.size(); }
  RigidTransform at(casadi_int index) const { return RigidTransform(T(t_(casadi::Slice(),index)),r_.at(index)); }
 private:
  Rotation<T> r_;
  T t_;
};
}
