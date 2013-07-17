#ifndef __NPEIGEN_HPP__
#define __NPEIGEN_HPP__

#include <Eigen/Core>
#include <boost/python.hpp>
#include <boost/format.hpp>
#include <string>

#include <iostream>

boost::python::object GetNumPyMod();

template<typename Scalar>
struct NPMatrixTypes {
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixType;
  typedef Eigen::Map<MatrixType> MapType;
  static const char* scalar_npname;
};

template<typename Scalar>
class NPMatrix : public NPMatrixTypes<Scalar>::MapType {
public:
  typedef typename NPMatrixTypes<Scalar>::MapType Base;

  NPMatrix() : Base(NULL, 0, 0) { }

  NPMatrix(int rows, int cols) : Base(NULL, 0, 0) {
    resize(rows, cols);
  }

  NPMatrix(const NPMatrix& other) : Base(NULL, 0, 0) {
    std::cout << "copy constructor " << other.transpose() << std::endl;
    // resize(other.rows(), other.cols());
    // Base::operator=(other);
    operator=(other);
  }

  template<typename OtherDerived>
  NPMatrix(const Eigen::MatrixBase<OtherDerived>& other) : Base(NULL, 0, 0) {
    // resize(other.rows(), other.cols());
    // Base::operator=(other);
    operator=(other);
  }

  // TODO
  NPMatrix(const boost::python::object &other) : Base(NULL, 0, 0) {
    operator=(Wrap(other));
  }


  static NPMatrix Wrap(const boost::python::object &other) {
    NPMatrix out;
    out.m_ndarray = other;
    std::string dtype = boost::python::extract<std::string>(other.attr("dtype").attr("name"));
    if (dtype != NPMatrixTypes<Scalar>::scalar_npname) {
      throw std::runtime_error((boost::format("Error converting Python ndarray to Eigen matrix: expected dtype %s, got %s instead")
        % NPMatrixTypes<Scalar>::scalar_npname % dtype).str());
    }
    boost::python::tuple shape = boost::python::extract<boost::python::tuple>(other.attr("shape"));
    switch (boost::python::len(shape)) {
    case 1:
      out.resetMap(boost::python::extract<int>(shape[0]), 1);
      break;
    case 2:
      out.resetMap(boost::python::extract<int>(shape[0]), boost::python::extract<int>(shape[1]));
      break;
    default:
      throw std::runtime_error("ndarray must have rank 1 or 2");
    }
    return out;
  }


  // Existing data will be wiped unless size is unchanged (same guarantee as Eigen::Matrix)
  void resize(int rows, int cols) {
    if (rows == this->rows() && cols == this->cols()) return;
    m_ndarray = makeNdarray(rows, cols);
    resetMap(rows, cols);
  }

  NPMatrix& operator=(const NPMatrix& other) {
    std::cout << "special operator=" << std::endl;
    resize(other.rows(), other.cols());
    Base::operator=(other);
    // std::cout << *this << std::endl;
    // std::cout << "is equal? " << this->m_ndarray.ptr() << ' ' << other.m_ndarray.ptr() << std::endl;
    return *this;
  }

  template<typename OtherDerived>
  NPMatrix& operator=(const Eigen::MatrixBase<OtherDerived>& other) {
    resize(other.rows(), other.cols());
    std::cout << "operator= " << other.transpose() << std::endl;
    Base::operator=(other);
    std::cout << "result= " << this->transpose() << std::endl;
    return *this;
  }

  NPMatrix& operator=(const boost::python::object &other) {
    return operator=(Wrap(other));
  }

  const boost::python::object &ndarray() const { return m_ndarray; }

private:
  boost::python::object m_ndarray;




  void resetMap(int rows, int cols) {
    new (this) Base(getNdarrayPointer(m_ndarray), rows, cols);
  }

  static boost::python::object makeNdarray(int rows, int cols) {
    return GetNumPyMod().attr("zeros")(boost::python::make_tuple(rows, cols), NPMatrixTypes<Scalar>::scalar_npname, "C");
  }

  static Scalar* getNdarrayPointer(const boost::python::object& arr) {
    return reinterpret_cast<Scalar*> (boost::python::extract<long int>(arr.attr("ctypes").attr("data"))());
  }
};

typedef NPMatrix<int> NPMatrixi;
typedef NPMatrix<float> NPMatrixf;
typedef NPMatrix<double> NPMatrixd;

#endif // __NPEIGEN_HPP__
