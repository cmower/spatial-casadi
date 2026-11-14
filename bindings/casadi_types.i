namespace casadi {
class SXElem {};
template<class T> class Matrix {};
%template(DM) Matrix<double>;
%template(SX) Matrix<SXElem>;
typedef Matrix<double> DM;
typedef Matrix<SXElem> SX;
class MX {};
}
