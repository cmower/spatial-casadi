#pragma once
#include <spatial_casadi/rigid_transform.hpp>

namespace spatial_casadi {
template<class T> T deg2rad(const T& x) { return x*(detail::pi/180); }
template<class T> T rad2deg(const T& x) { return x*(180/detail::pi); }

/// Compatibility wrapper for a translation vector.
template<class T> class Translation {
 public:
  explicit Translation(const T& vector) : t_(vector) { detail::shape(t_,3); }
  T as_vector() const { return t_; }
  T magnitude() const { return detail::columns(t_,[](const T& v) { return norm_2(v); }); }
  Translation add(const Translation& other) const {
    detail::count(t_.size2(),other.t_.size2());
    return Translation(t_+other.t_);
  }
  Translation neg() const { return Translation(-t_); }
  RigidTransform<T> as_transform() const { return RigidTransform<T>::from_translation(t_); }
 private:
  T t_;
};
}
