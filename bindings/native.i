%module(package="spatial_casadi") native
%{
#include <spatial_casadi/rotation.hpp>
%}
typedef long long int casadi_int;
%include <std_string.i>
%include <exception.i>
%exception {
  try { $action }
  catch (const std::exception& e) { SWIG_exception(SWIG_ValueError, e.what()); }
}
%import(module="casadi") "casadi_types.i"
%include "spatial_casadi/rotation.hpp"
%template(RotationDM) spatial_casadi::Rotation<casadi::DM>;
%template(RotationSX) spatial_casadi::Rotation<casadi::SX>;
%template(RotationMX) spatial_casadi::Rotation<casadi::MX>;
