#ifndef __NPEIGEN_HPP__
#define __NPEIGEN_HPP__

#include <Eigen/Core>
#include <boost/python.hpp>
#include <boost/format.hpp>
#include <string>

#include <iostream>

namespace py = boost::python;
py::object GetNumPyMod();

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

  NPMatrix() : m_initialized(false), Base(NULL, 0, 0) { }

  NPMatrix(int rows, int cols) : m_initialized(false), Base(NULL, 0, 0) {
    resize(rows, cols);
    assert(m_initialized);
  }

  NPMatrix(const NPMatrix& other) : m_initialized(false), Base(NULL, 0, 0) {
    assert(other.initialized());
    operator=(other);
    assert(m_initialized);
  }

  template<typename OtherDerived>
  NPMatrix(const Eigen::MatrixBase<OtherDerived>& other) : m_initialized(false), Base(NULL, 0, 0) {
    operator=(other);
    assert(m_initialized);
  }

  NPMatrix(const py::object &other) : m_initialized(false), Base(NULL, 0, 0) {
    operator=(Wrap(other));
    assert(m_initialized);
  }


  static NPMatrix Wrap(const py::object &other) {
    py::object in = other;
    std::string dtype = py::extract<std::string>(other.attr("dtype").attr("name"));
    std::string target_dtype(NPMatrixTypes<Scalar>::scalar_npname);
    if (dtype != target_dtype) {
      // possible upcasting from int to float
      if ((target_dtype == "float32" || target_dtype == "float64") && (dtype == "int32" || dtype == "int64")) {
        in = other.attr("astype")(target_dtype);
      } else {
        throw std::runtime_error((boost::format("Error converting Python ndarray to Eigen matrix: expected dtype %s, got %s instead")
          % NPMatrixTypes<Scalar>::scalar_npname % dtype).str());
      }
    }

    NPMatrix out;
    out.m_ndarray = in;
    py::tuple shape = py::extract<py::tuple>(in.attr("shape"));
    switch (py::len(shape)) {
    case 1:
      out.resetMap(getNdarrayPointer(out.m_ndarray), py::extract<int>(shape[0]), 1);
      break;
    case 2:
      out.resetMap(getNdarrayPointer(out.m_ndarray), py::extract<int>(shape[0]), py::extract<int>(shape[1]));
      break;
    default:
      throw std::runtime_error("ndarray must have rank 1 or 2");
    }
    assert(out.initialized());
    return out;
  }


  // Existing data will be wiped unless size is unchanged (same guarantee as Eigen::Matrix)
  void resize(int rows, int cols) {
    if (rows == this->rows() && cols == this->cols()) return;
    m_ndarray = makeNdarray(rows, cols);
    resetMap(getNdarrayPointer(m_ndarray), rows, cols);
    assert(m_initialized);
  }

  NPMatrix& operator=(const NPMatrix& other) {
    assert(other.initialized());
    resize(other.rows(), other.cols());
    Base::operator=(other);
    return *this;
  }

  template<typename OtherDerived>
  NPMatrix& operator=(const Eigen::MatrixBase<OtherDerived>& other) {
    resize(other.rows(), other.cols());
    Base::operator=(other);
    return *this;
  }

  NPMatrix& operator=(const py::object &other) {
    return operator=(Wrap(other));
  }

  const py::object &ndarray() const { return m_ndarray; }
  bool initialized() const { return m_initialized; }


  // Boost Python converters
  struct ToPythonConverter {
    static PyObject* convert(const NPMatrix<Scalar>& m) {
      return py::incref(m.ndarray().ptr());
    }
  };
  struct FromPythonConverter {
    FromPythonConverter() {
      py::converter::registry::push_back(&convertible, &construct, py::type_id<NPMatrix<Scalar> >());
    }
    static void* convertible(PyObject* obj_ptr) {
      // leave the check to the NPMatrix constructor
      return obj_ptr;
    }
    static void construct(PyObject* obj_ptr, py::converter::rvalue_from_python_stage1_data* data) {
      py::object value(py::handle<>(py::borrowed(obj_ptr)));
      void* storage = ((py::converter::rvalue_from_python_storage<NPMatrix<Scalar> >*) data)->storage.bytes;
      new (storage) NPMatrix<Scalar>(value);
      data->convertible = storage;
    }
  };

private:
  py::object m_ndarray;
  bool m_initialized; // true if the underlying Eigen Map's pointer points to something non-null

  void resetMap(Scalar* storage, int rows, int cols) {
    new (this) Base(storage, rows, cols);
    m_initialized = storage != NULL;
  }

  static py::object makeNdarray(int rows, int cols) {
    return GetNumPyMod().attr("zeros")(py::make_tuple(rows, cols), NPMatrixTypes<Scalar>::scalar_npname, "C");
  }

  static Scalar* getNdarrayPointer(const py::object& arr) {
    return reinterpret_cast<Scalar*> (py::extract<long int>(arr.attr("ctypes").attr("data"))());
  }
};

typedef NPMatrix<int> NPMatrixi;
typedef NPMatrix<float> NPMatrixf;
typedef NPMatrix<double> NPMatrixd;


// Utility Boost Python converters for general Eigen matrices
// these converters make copies in both directions (C++ <--> Python)
template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
struct EigenMatrixConverters {
  typedef Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols> MatrixType;

  struct ToPython {
    static PyObject* convert(const MatrixType& m) {
      return py::incref(NPMatrix<_Scalar>(m).ndarray().ptr());
    }
  };


  struct FromPython {
    FromPython() {
      py::converter::registry::push_back(&convertible, &construct, py::type_id<MatrixType>());
    }
    static void* convertible(PyObject* obj_ptr) {
      // leave the check to the NPMatrix constructor
      return obj_ptr;
    }
    static void construct(PyObject* obj_ptr, py::converter::rvalue_from_python_stage1_data* data) {
      py::object value(py::handle<>(py::borrowed(obj_ptr)));
      void* storage = ((py::converter::rvalue_from_python_storage<MatrixType>*) data)->storage.bytes;
      new (storage) MatrixType(NPMatrix<_Scalar>::Wrap(value));
      data->convertible = storage;
    }
  };
};



#endif // __NPEIGEN_HPP__
