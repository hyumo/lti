
# LTI

[WIP] An header only C++17 library for linear time invariant system based on Eigen3

## Usage
Just include the `lti.hpp` in your project.

```
#include <lti.hpp>

int main()
{
    int nx = 2, nu = 2, ny = 2;
    Eigen::MatrixXd A(nx, nx);
    Eigen::MatrixXd B(nx, nu);
    Eigen::MatrixXd C(ny, nx);
    Eigen::MatrixXd D(ny, nu);
    A.setRandom();
    B.setRandom();
    C.setIdentity();
    D.setZero();

    //----------------------------------------
    // Continuous LTI state space representation
    // der(x) = Ax + Bu;
    //      y = Cx + Du;
    //----------------------------------------
    lti::css sys_c(A, B, C, D);

    //----------------------------------------
    // Discrete LTI state space representation
    // x(k+1) = A(k) + Bu(k);
    //   y(k) = Cx(k) + D(k);
    //----------------------------------------
    double dt = 0.001;
    lti::dss sys_d(A, B, C, D, dt);

    //---------------------------------
    // Create a transfer function
    //          s + 2
    // Y/U = -----------
    //       s^2 + 2s + 3
    //---------------------------------
    Eigen::ArrayXd num(2);
    Eigen::ArrayXd den(3);
    num << 1, 2;
    den << 1, 2, 3;
    lti::tf sys3(num, den);
}
```
