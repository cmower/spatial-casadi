#include <iostream>
#include <spatial_casadi/spatial.hpp>

int main() {
  using casadi::SX;
  SX x = SX::sym("x", 6);
  auto pose = spatial_casadi::RigidTransform<SX>::from_exp_coords(x);
  SX point = pose.apply(SX(std::vector<double>{1, 0, 0}));
  casadi::Function f("point", {x}, {point, SX::jacobian(point, x)});
  for (const auto& value : f(std::vector<casadi::DM>{casadi::DM::zeros(6)}))
    std::cout << value << '\n';
}
