#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>

#include <limits>
#include <stdexcept>

namespace lti
{
namespace details
{
class stateSpace
{
protected:
    Eigen::Index _nx;
    Eigen::Index _nu;
    Eigen::Index _ny;

    Eigen::MatrixXd _A;
    Eigen::MatrixXd _B;
    Eigen::MatrixXd _C;
    Eigen::MatrixXd _D;

    Eigen::VectorXd _x;

public:
    stateSpace() = delete;

    stateSpace(Eigen::MatrixXd A)
        : stateSpace(A, Eigen::MatrixXd::Zero(A.cols(), 0))
    {
    }

    stateSpace(Eigen::MatrixXd A, Eigen::MatrixXd B)
        : stateSpace(A, B, Eigen::MatrixXd::Identity(A.cols(), A.cols()))
    {
    }

    stateSpace(Eigen::MatrixXd A, Eigen::MatrixXd B, Eigen::MatrixXd C)
        : stateSpace(A, B, C, Eigen::MatrixXd::Zero(C.rows(), B.cols()))
    {
    }

    stateSpace(Eigen::MatrixXd A, Eigen::MatrixXd B, Eigen::MatrixXd C,
               Eigen::MatrixXd D)
        : _A(A), _B(B), _C(C), _D(D), _nx(A.cols()), _nu(B.cols()),
          _ny(C.rows()), _x(A.cols())
    {
        if (_x.size() != _A.rows()) {
            throw std::invalid_argument("A must be a square matrix");
        }
        if (_x.size() != _B.rows()) {
            throw std::invalid_argument(
                "A and B must have the same number of rows.");
        }
        if (_x.size() != _C.cols()) {
            throw std::invalid_argument(
                "A and C must have the same number of columns.");
        }
        if (_B.cols() != _D.cols()) {
            throw std::invalid_argument(
                "B and D must have the same number of columns.");
        }
        if (_C.rows() != _D.rows()) {
            throw std::invalid_argument(
                "C and D must have the same number of rows.");
        }
    }

    bool isSISO() const
    {
        return _nu == 1 && _ny == 1;
    }

    /**
     * \brief Return the eigehn values of the state space system
     */
    auto eigenValues() const
    {
        if (_nx == 0) {
            throw std::runtime_error(
                "Not able to get poles with empty A matrix");
        }
        return _A.eigenvalues();
    }

    /**
     * \brief Controlability matrix
     */
    auto ctrl() const
    {
        if (_nx == 0) {
            throw std::runtime_error(
                "Not able to get controlability matrix with 0 state");
        }
        if (_nu == 0) {
            throw std::runtime_error(
                "Not able to get controlability matrix with 0 input");
        }
        Eigen::MatrixXd cm(_nx, _nx * _nu);
        cm.block(0, 0, _nx, _nu) = _B;
        for (auto i = 1; i < _nx; ++i) {
            cm.block(0, i * _nu, _nx, _nu)
                = _A * cm.block(0, (i - 1LL) * _nu, _nx, _nu);
        }
        return cm;
    }

    /**
     * \brief observability matrix
     */
    auto obsv() const
    {
        if (_nx == 0) {
            throw std::runtime_error(
                "Not able to get observability matrix with 0 state");
        }
        if (_ny == 0) {
            throw std::runtime_error(
                "Not able to get observability matrix with 0 output");
        }
        Eigen::MatrixXd om(_nx * _ny, _nx);
        om.block(0, 0, _ny, _nx) = _C;
        for (auto i = 1; i < _nx; ++i) {
            om.block(i * _ny, 0, _ny, _nx)
                = om.block((i - 1LL) * _ny, 0, _ny, _nx) * _A;
        }
        return om;
    }
};

class polynomial
{
private:
    /**
     * \brief Polynomial coefficients from higher to lower order. Last element
     * is the zero-th order coefficient.
     */
    Eigen::VectorXd _c;

public:
    polynomial() = delete;

    polynomial(Eigen::VectorXd c) : _c(c) {}

    auto eval(double x) const
    {
        auto y = _c[0];
        for (auto i = 1; i < _c.size(); ++i) {
            y = _c[i] + x * y;
        }
        return y;
    }
};

} // namespace details

/**
 * \brief Transfer function representation
 */
class tf
{
private:
    Eigen::VectorXd _n;
    Eigen::VectorXd _d;

public:
    tf() = delete;
    tf(Eigen::VectorXd n, Eigen::VectorXd d) : _n(n), _d(d) {}
};

/**
 * \brief Contiuous state space representation
 */
class css : public details::stateSpace
{
private:
    /* data */
public:
    css(Eigen::MatrixXd A) : stateSpace(A) {}

    css(Eigen::MatrixXd A, Eigen::MatrixXd B) : stateSpace(A, B) {}

    css(Eigen::MatrixXd A, Eigen::MatrixXd B, Eigen::MatrixXd C)
        : stateSpace(A, B, C)
    {
    }

    css(Eigen::MatrixXd A, Eigen::MatrixXd B, Eigen::MatrixXd C,
        Eigen::MatrixXd D)
        : stateSpace(A, B, C, D)
    {
    }

    Eigen::MatrixXd dcGain() const
    {
        // return a zero matrix if there no input or output
        if (_nu == 0 || _ny == 0) {
            return Eigen::MatrixXd::Zero(_ny, _nu);
        }

        // solve 0 = Ax + B*1 to find steady-state x = -inv(A)B;
        Eigen::MatrixXd x = -_A.fullPivLu().solve(_B);

        // check if there's a solution by validating the norm of ||Ax+B||
        if ((_A * x + _B).norm()
            > 100 * std::numeric_limits<double>::epsilon()) {
            // No solution
            Eigen::MatrixXd r(_ny, _nu);
            r.setConstant(std::numeric_limits<double>::infinity());
            return r;
        }
        return _D + _C * x;
    }

    css(/* args */) = delete;
};

/**
 * \brief Discrete state space representation
 */
class dss : public details::stateSpace
{
private:
    /* data */
public:
    dss(/* args */) = delete;
};
} // namespace lti
