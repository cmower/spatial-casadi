#pragma once
#include <casadi/casadi.hpp>
#include <cmath>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace spatial_casadi::detail {
inline constexpr double pi = 3.14159265358979323846;
inline void require(bool valid, const char* message) {
  if (!valid) throw std::invalid_argument(message);
}
template<class T> T cat(std::initializer_list<T> values) {
  return T::vertcat(std::vector<T>(values));
}
template<class T> T col(const T& x, casadi_int i) {
  return x(casadi::Slice(), x.size2() == 1 ? 0 : i);
}
inline casadi_int count(casadi_int a, casadi_int b) {
  require(a == b || a == 1 || b == 1, "Batch sizes must match or be one.");
  return std::max(a, b);
}
template<class T, class F> T columns(const T& x, F f) {
  std::vector<T> result;
  for (casadi_int i = 0; i < x.size2(); ++i) result.push_back(f(col(x, i)));
  return T::horzcat(result);
}
template<class T> void shape(const T& x, casadi_int rows) {
  require(x.size1() == rows && x.size2() > 0, "Unexpected matrix shape or empty batch.");
  if constexpr (std::is_same_v<T, casadi::DM>)
    require(x.is_regular(), "Inputs must be finite.");
}
template<class T> T normalize(const T& q) {
  T s = dot(q, q);
  if constexpr (std::is_same_v<T, casadi::DM>)
    require(static_cast<double>(s) > 0, "Quaternion must be nonzero.");
  return q / sqrt(s);
}
template<class T> T canonical(const T& q) {
  T negative = (q(3) < 0) + (q(3) == 0) *
    ((q(0) < 0) + (q(0) == 0) * ((q(1) < 0) + (q(1) == 0) * (q(2) < 0)));
  return T::if_else(negative > 0, -q, q);
}
template<class T> T multiply(const T& a, const T& b) {
  T av = a(casadi::Slice(0, 3)), bv = b(casadi::Slice(0, 3));
  return cat<T>({a(3)*bv + b(3)*av + cross(av, bv), a(3)*b(3) - dot(av, bv)});
}
template<class T> T conjugate(const T& q) {
  return cat<T>({-q(casadi::Slice(0, 3)), q(3)});
}
template<class T> T rotate(const T& q, const T& v) {
  T u = q(casadi::Slice(0, 3));
  T uv = cross(u, v);
  return v + 2*q(3)*uv + 2*cross(u, uv);
}
template<class T> T exp_quat(const T& v) {
  T s = dot(v, v), small = s < 1e-8;
  // Guard the general branch too: AD can evaluate inactive expressions.
  T a = sqrt(T::if_else(small, T(1), s));
  T scale = T::if_else(small, 0.5-s/48+s*s/3840-s*s*s/645120, sin(a/2)/a);
  T w = T::if_else(small, 1-s/8+s*s/384-s*s*s/46080, cos(a/2));
  return cat<T>({scale*v, w});
}
template<class T> T log_quat(const T& input) {
  T q = canonical(input), v = q(casadi::Slice(0, 3));
  T s = dot(v, v), small = s < 1e-8;
  T a = sqrt(T::if_else(small, T(1), s));
  T scale = T::if_else(small, 2+s/3+3*s*s/20+5*s*s*s/56, 2*atan2(a, q(3))/a);
  return scale*v;
}
inline bool euler_sequence(const std::string& seq, bool output = false) {
  require(!seq.empty() && seq.size() <= 3 && (!output || seq.size() == 3),
          "Euler sequence must have one to three axes (three for output).");
  bool intrinsic = seq[0] >= 'X' && seq[0] <= 'Z';
  for (size_t i = 0; i < seq.size(); ++i) {
    require(intrinsic ? seq[i] >= 'X' && seq[i] <= 'Z' : seq[i] >= 'x' && seq[i] <= 'z',
            "Use only xyz or XYZ axes; do not mix cases.");
    require(i == 0 || seq[i] != seq[i-1], "Adjacent Euler axes must differ.");
  }
  return intrinsic;
}
inline int axis(char c) { return (c >= 'X' && c <= 'Z') ? c-'X' : c-'x'; }
template<class T> T matrix_quat(const T& m) {
  T trace = m(0,0)+m(1,1)+m(2,2);
  T q = cat<T>({m(2,1)-m(1,2), m(0,2)-m(2,0), m(1,0)-m(0,1), 1+trace});
  T best = trace;
  for (int i = 0; i < 3; ++i) {
    int j = (i+1)%3, k = (i+2)%3;
    T alt = T::zeros(4,1);
    alt(i)=1-trace+2*m(i,i); alt(j)=m(j,i)+m(i,j);
    alt(k)=m(k,i)+m(i,k); alt(3)=m(k,j)-m(j,k);
    q = T::if_else(m(i,i) > best, alt, q);
    best = fmax(best, m(i,i));
  }
  return normalize(q);
}
}
