#include <iostream>
#include <spatial_casadi/spatial.hpp>

using casadi::DM;
using namespace spatial_casadi;

void close(const DM& actual, const DM& expected, double tolerance = 1e-12) {
  if (!actual.is_regular() || static_cast<double>(norm_fro(actual - expected)) > tolerance)
    throw std::runtime_error("C++ numerical check failed");
}

template <class T>
void symbolic_checks() {
  T x = T::sym("x", 6);
  auto transform = RigidTransform<T>::from_exp_coords(x);
  T coordinates = transform.as_exp_coords();
  T cost = dot(T::ones(6, 1), coordinates);
  casadi::Function f("f", {x}, {coordinates, T::jacobian(coordinates, x), T::hessian(cost, x)});
  auto values = f(std::vector<DM>{DM::zeros(6)});
  close(values[0], DM::zeros(6));
  close(values[1], DM::eye(6));
  close(values[2], DM::zeros(6, 6));
  T matrix = RigidTransform<T>::from_matrix(transform.as_matrix()).as_matrix();
  casadi::Function g("g", {x}, {matrix});
  close(g(std::vector<DM>{DM::zeros(6)})[0], DM::eye(4));
}

int main() {
  try {
    symbolic_checks<casadi::SX>();
    symbolic_checks<casadi::MX>();
    auto r = Rotation<DM>::from_euler("z", DM(90), true);
    close(r.apply(DM{1, 0, 0}), DM{0, 1, 0});
    auto tf = RigidTransform<DM>::from_components(DM{1, 2, 3}, r);
    close(tf.apply(DM{1, 0, 0}), DM{1, 3, 3});
    close((tf * tf.inv()).as_matrix(), DM::eye(4));
    close(RigidTransform<DM>::from_dual_quat(tf.as_dual_quat()).as_matrix(), tf.as_matrix());
    close(RigidTransform<DM>::from_exp_coords(tf.as_exp_coords()).as_matrix(), tf.as_matrix());
    auto rotations = Rotation<DM>::concatenate({Rotation<DM>::identity(), r});
    Slerp<DM> interpolation(DM{0, 1}, rotations);
    close(interpolation.evaluate(DM(0.5)).as_matrix(), r.power(0.5).as_matrix());
    auto transforms = RigidTransform<DM>::concatenate({tf, tf.inv()});
    close(transforms.at(1).as_matrix(), tf.inv().as_matrix());
    close(Rotation<DM>::random(3, 42).as_quat(), Rotation<DM>::random(3, 42).as_quat());
    std::cout << "C++ transforms, batches, interpolation and derivatives passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
