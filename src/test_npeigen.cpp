#include <iostream>
#include "npeigen.hpp"

using namespace std;

template<typename T>
void eigen_func(const Eigen::MatrixBase<T> &m) {
  cout << m*2 << endl;
}

int main() {
  Py_Initialize();

  int rows = 4, cols = 5;
  NPMatrixd n;
  n.resize(10, 10);
  n.setZero();
  cout << n << endl;



  NPMatrixd m(rows, cols);
  cout << "Initial matrix (uninitialized memory)\n" << m << '\n' << endl;

  cout << "Testing setting from Python/NumPy (entries should be 0, 1, 2, ... row-wise):\n";
  int c = 0;
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      m.ndarray()[i][j] = c++;
    }
  }
  cout << m << '\n' << endl;

  cout << "Testing setting from C++/Eigen (entries should be same as above, but negated):\n";
  m = -m;
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      cout << boost::python::extract<double>(m.ndarray()[i][j]) << ' ';
    }
    cout << '\n';
  }
  cout << endl;

  cout << "Testing C++ function taking Eigen::MatrixBase (entries should be double above):\n";
  eigen_func(m);

  return 0;
}
