#include <spatial_casadi/rotation.hpp>
#include <iostream>

int main() {
  using casadi::SX;
  using spatial_casadi::Rotation;
  SX x=SX::sym("x",3);
  SX q=Rotation<SX>::from_rotvec(x).as_quat();
  casadi::Function f("f",{x},{q,SX::jacobian(q,x),SX::hessian(q(3),x)});
  auto values=f(std::vector<casadi::DM>{casadi::DM::zeros(3)});
  for (const auto& value: values) if (!value.is_regular()) return 1;
  auto identity=Rotation<casadi::DM>::from_matrix(casadi::DM::eye(3)).as_matrix();
  if (static_cast<double>(norm_fro(identity-casadi::DM::eye(3))) > 1e-12) return 2;
  std::cout << "C++ rotation and derivatives passed\n";
}
