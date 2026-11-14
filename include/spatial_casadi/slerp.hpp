#pragma once
#include <spatial_casadi/rotation.hpp>

namespace spatial_casadi {
/// Piecewise shortest-path spherical interpolation over fixed knot times.
template <class T>
class Slerp {
 public:
  Slerp(const casadi::DM& times, const Rotation<T>& rotations)
      : times_(times), rotations_(rotations) {
    detail::require(times.size1() == 1 || times.size2() == 1, "Times must be a vector.");
    detail::require(times.is_regular() && times.numel() == rotations.size() && times.numel() >= 2,
                    "Provide at least two finite times, one per rotation.");
    for (casadi_int i = 1; i < times.numel(); ++i)
      detail::require(static_cast<double>(times(i)) > static_cast<double>(times(i - 1)),
                      "Times must be strictly increasing.");
  }
  Rotation<T> evaluate(const T& times) const {
    detail::shape(times, 1);
    std::vector<T> result;
    for (casadi_int j = 0; j < times.numel(); ++j) {
      T t = times(j);
      if constexpr (std::is_same_v<T, casadi::DM>)
        detail::require(
            static_cast<double>(t) >= static_cast<double>(times_(0)) &&
                static_cast<double>(t) <= static_cast<double>(times_(times_.numel() - 1)),
            "Interpolation time is outside the knot interval.");
      T q;
      for (casadi_int i = 0; i < rotations_.size() - 1; ++i) {
        auto start = rotations_.at(i);
        T delta = start.inv().compose(rotations_.at(i + 1)).as_rotvec();
        double lo = static_cast<double>(times_(i)), hi = static_cast<double>(times_(i + 1));
        T next = start.compose(Rotation<T>::from_rotvec(delta * ((t - lo) / (hi - lo)))).as_quat();
        q = i == 0 ? next : T::if_else(t >= lo, next, q);
      }
      result.push_back(q);
    }
    return Rotation<T>(T::horzcat(result), false);
  }

 private:
  casadi::DM times_;
  Rotation<T> rotations_;
};
}  // namespace spatial_casadi
