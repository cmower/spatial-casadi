%module(package="spatial_casadi") native
%{
#include <spatial_casadi/spatial.hpp>
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

%include "spatial_casadi/rigid_transform.hpp"
%template(RigidTransformDM) spatial_casadi::RigidTransform<casadi::DM>;
%template(RigidTransformSX) spatial_casadi::RigidTransform<casadi::SX>;
%template(RigidTransformMX) spatial_casadi::RigidTransform<casadi::MX>;

%include "spatial_casadi/slerp.hpp"
%template(SlerpDM) spatial_casadi::Slerp<casadi::DM>;
%template(SlerpSX) spatial_casadi::Slerp<casadi::SX>;
%template(SlerpMX) spatial_casadi::Slerp<casadi::MX>;

%include "spatial_casadi/translation.hpp"
%template(TranslationDM) spatial_casadi::Translation<casadi::DM>;
%template(TranslationSX) spatial_casadi::Translation<casadi::SX>;
%template(TranslationMX) spatial_casadi::Translation<casadi::MX>;
%template(deg2radDM) spatial_casadi::deg2rad<casadi::DM>;
%template(deg2radSX) spatial_casadi::deg2rad<casadi::SX>;
%template(deg2radMX) spatial_casadi::deg2rad<casadi::MX>;
%template(rad2degDM) spatial_casadi::rad2deg<casadi::DM>;
%template(rad2degSX) spatial_casadi::rad2deg<casadi::SX>;
%template(rad2degMX) spatial_casadi::rad2deg<casadi::MX>;
