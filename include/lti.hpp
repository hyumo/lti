#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>

#include <algorithm>
#include <complex>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace lti
{
namespace details
{
template <bool isDiscrete>
class stateSpace
{
protected:
    Eigen::Index _nx; /**< Number of states */
    Eigen::Index _nu; /**< Number of sys inputs */
    Eigen::Index _ny; /**< Number of sys outputs */

    Eigen::MatrixXd _A;
    Eigen::MatrixXd _B;
    Eigen::MatrixXd _C;
    Eigen::MatrixXd _D;

    Eigen::VectorXd _x;

    double _dt;

public:
    stateSpace() = delete;

    stateSpace(Eigen::MatrixXd A, double dt = 0.0)
        : stateSpace(A, Eigen::MatrixXd::Zero(A.cols(), 0), dt)
    {
    }

    stateSpace(Eigen::MatrixXd A, Eigen::MatrixXd B, double dt = 0.0)
        : stateSpace(A, B, Eigen::MatrixXd::Identity(A.cols(), A.cols()), dt)
    {
    }

    stateSpace(Eigen::MatrixXd A, Eigen::MatrixXd B, Eigen::MatrixXd C,
               double dt = 0.0)
        : stateSpace(A, B, C, Eigen::MatrixXd::Zero(C.rows(), B.cols()), dt)
    {
    }

    stateSpace(Eigen::MatrixXd A, Eigen::MatrixXd B, Eigen::MatrixXd C,
               Eigen::MatrixXd D, double dt = 0.0)
        : _A(A), _B(B), _C(C), _D(D), _nx(A.cols()), _nu(B.cols()),
          _ny(C.rows()), _x(A.cols()), _dt(dt)
    {
        if constexpr (isDiscrete) {
            if (fabs(dt) < 100 * std::numeric_limits<double>::epsilon()) {
                throw std::invalid_argument(
                    "dt should be greater than zero for a discrete system");
            }
        } else {
            if (fabs(dt) > 100 * std::numeric_limits<double>::epsilon()) {
                throw std::invalid_argument(
                    "dt must be zero for a continuous system");
            }
        }

        if (_x.size() != A.rows()) {
            throw std::invalid_argument("A must be a square matrix");
        }
        if (_x.size() != B.rows()) {
            throw std::invalid_argument(
                "A and B must have the same number of rows.");
        }
        if (_x.size() != C.cols()) {
            throw std::invalid_argument(
                "A and C must have the same number of columns.");
        }
        if (B.cols() != D.cols()) {
            throw std::invalid_argument(
                "B and D must have the same number of columns.");
        }
        if (C.rows() != D.rows()) {
            throw std::invalid_argument(
                "C and D must have the same number of rows.");
        }
    }

    friend std::ostream &operator<<(std::ostream &os, const stateSpace &ss)
    {
        os << "A(" << ss._A.rows() << ", " << ss._A.cols() << "):\n"
           << ss._A << "\n";
        os << "B(" << ss._B.rows() << ", " << ss._B.cols() << "):\n"
           << ss._B << "\n";
        os << "C(" << ss._C.rows() << ", " << ss._C.cols() << "):\n"
           << ss._C << "\n";
        os << "D(" << ss._D.rows() << ", " << ss._D.cols() << "):\n"
           << ss._D << "\n";
        return os;
    }

    const Eigen::MatrixXd &A() const noexcept
    {
        return _A;
    }
    const Eigen::MatrixXd &B() const noexcept
    {
        return _B;
    }
    const Eigen::MatrixXd &C() const noexcept
    {
        return _C;
    }
    const Eigen::MatrixXd &D() const noexcept
    {
        return _D;
    }

    /**
     * \brief Negate a system
     */
    stateSpace operator-() const noexcept
    {
        return stateSpace(_A, _B, -_C, -_D, _dt);
    }

    stateSpace operator+(const stateSpace &rhs)
    {
        if (_B.cols() != rhs._B.cols() || _C.rows() != rhs._C.rows()) {
            throw std::invalid_argument("Cannot add two system with different"
                                        "input or output dimensions.");
        }

        if constexpr (isDiscrete) {
            if (fabs(_dt - rhs._dt)
                > 100 * std::numeric_limits<double>::epsilon()) {
                throw std::invalid_argument(
                    "Cannot add two system with different sampling time.");
            }
        }

        const Eigen::MatrixXd &A1 = _A;
        const Eigen::MatrixXd &B1 = _B;
        const Eigen::MatrixXd &C1 = _C;
        const Eigen::MatrixXd &D1 = _D;
        const Eigen::MatrixXd &A2 = rhs._A;
        const Eigen::MatrixXd &B2 = rhs._B;
        const Eigen::MatrixXd &C2 = rhs._C;
        const Eigen::MatrixXd &D2 = rhs._D;

        Eigen::MatrixXd A(A1.rows() + A2.rows(), A1.cols() + A2.cols());
        Eigen::MatrixXd B(B1.rows() + B2.rows(), B1.cols());
        Eigen::MatrixXd C(C1.rows(), C1.cols() + C2.cols());
        Eigen::MatrixXd D(D1.rows(), D1.cols());

        // A = [A1, zeros(nx1, nx2); zeros(nx2, nx1), A2]
        A.setZero();
        A.block(0, 0, A1.rows(), A1.cols()) = A1;
        A.block(A1.rows(), A1.cols(), A2.rows(), A2.cols()) = A2;

        // B = [B1;B2]
        B.block(0, 0, B1.rows(), B1.cols()) = B1;
        B.block(B1.rows(), 0, B2.rows(), B2.cols()) = B2;

        // C = [C1,C2]
        C.block(0, 0, C1.rows(), C1.cols()) = C1;
        C.block(0, C1.cols(), C2.rows(), C2.cols()) = C2;

        // _D = D1+D2
        D = D1 + D2;

        return stateSpace(A, B, C, D, _dt);
    }

    stateSpace operator-(const stateSpace &rhs)
    {
        return -rhs + *this;
    }

    /**
     * \brief Check if two ss sytem are equal
     */
    bool operator==(const stateSpace &rhs) const noexcept
    {
        if constexpr (isDiscrete) {
            if (fabs(_dt - rhs._dt)
                > 100 * std::numeric_limits<double>::epsilon()) {
                return false;
            }
        }
        // a lambda to check if two matrices are equal
        auto isEqual
            = [](const Eigen::MatrixXd &m1, const Eigen::MatrixXd &m2) {
                  return m1.rows() != m2.rows() || m1.cols() != m2.cols()
                         || m1.isApprox(m2);
              };
        return isEqual(_A, rhs._A) && isEqual(_B, rhs._B) && isEqual(_C, rhs._C)
               && isEqual(_D, rhs._D);
    }

    bool operator!=(const stateSpace &rhs) const noexcept
    {
        return !(*this == rhs);
    }

    /**
     * \brief Check if the system is single input and single output
     */
    bool isSISO() const noexcept
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

    /**
     * \brief DC Gain of a continuous state system
     *
     * If no steady state solution is found, an exception is thrown.
     */
    Eigen::MatrixXd dcGain() const
    {
        // return a zero matrix if there no input or output
        if (_nu == 0 || _ny == 0) {
            return Eigen::MatrixXd::Zero(_ny, _nu);
        }

        if constexpr (isDiscrete) {
            throw std::runtime_error("Not implemented");
        } else {
            // solve 0 = Ax + B*1 to find steady-state x = -inv(A)B;
            Eigen::MatrixXd x = -_A.fullPivLu().solve(_B);

            // check if there's a solution by validating the norm of ||Ax+B||
            if ((_A * x + _B).norm()
                > 100 * std::numeric_limits<double>::epsilon()) {
                // No solution
                // Eigen::MatrixXd r(ny, nu);
                // r.setConstant(std::numeric_limits<double>::infinity());
                // return r;
                // TODO: Should I return an infinity matrix?
                throw std::runtime_error(
                    "Could not find dc gain. A might be singular.");
            }
            return _D + _C * x;
        }
    }
};

class polynomial
{
private:
    /**
     * \brief Polynomial coefficients from higher to lower order. Last element
     * is the zero-th order coefficient.
     */
    Eigen::ArrayXd _c;

public:
    polynomial() = delete;

    /**
     * \brief Polynomial constructor by an eigen array representing its
     * coefficents.
     *
     * Coefficents are from higher order to 0th order
     *
     * Example:
     * - c = [1] => y = 1.0;
     * - c = [1,2] => y = x + 2;
     *
     * Exception:
     * - `c = [0,1]` is not allowed, because it is in fact a 0th order poly but
     * using an array with 2 elements.
     */
    polynomial(Eigen::ArrayXd c) : _c(c)
    {
        if (_c.size() > 1) {
            if (std::abs(_c[0])
                < 100 * std::numeric_limits<double>::epsilon()) {
                throw std::invalid_argument(
                    "First element of the coefficent array cannot be zero");
            }
        }
    }

    /**
     * \brief Standard output for polynomial
     *
     * Print a polynomial up to 5th order. If the polynomial is greater than 5th
     * order, print the first 3 and last 2 elements
     */
    friend std::ostream &operator<<(std::ostream &os, const polynomial &p)
    {
        if (p._c.size() < 6) {
            os << p._c[0] << "s^" << p._c.size() - 1;
            for (Eigen::Index i = 1; i < p._c.size(); ++i) {
                os << ((p._c[i] > 0) ? " + " : " - ");
                os << std::abs(p._c[i]) << "s^" << p._c.size() - 1 - i;
            }
        } else {
            os << p._c[0] << "s^" << p._c.size() - 1;
            for (Eigen::Index i = 1; i < 3; ++i) {
                os << ((p._c[i] > 0) ? " + " : " - ");
                os << std::abs(p._c[i]) << "s^" << p._c.size() - 1 - i;
            }
            os << "... + " << p._c[p._c.size() - 2] << "s^1 + "
               << p._c[p._c.size() - 1] << "\n";
        }
        return os;
    }

    /**
     * \brief Evaluate a complex value
     */
    std::complex<double> eval(std::complex<double> x) const noexcept
    {
        auto y = std::complex(_c[0], 0.0);
        for (auto i = 1; i < _c.size(); ++i) {
            y = std::complex(_c[i], 0.0) + x * y;
        }
        return y;
    }

    /**
     * \brief Evaluate a real value
     */
    double eval(double x) const noexcept
    {
        auto y = _c[0];
        for (auto i = 1; i < _c.size(); ++i) {
            y = _c[i] + x * y;
        }
        return y;
    }

    /**
     * \brief Check if 2 polynomials are the same by checking the equality of
     * their coefficents
     *
     * Noted that this only checks poly coefficients.
     * Examples:
     * true == (poly([2, 1]) == poly([2, 1]))
     * false == (poly([0,1]) == poly([0,0,1]))
     */
    bool operator==(const polynomial &rhs) const
    {
        if (this->_c.size() != rhs._c.size()) {
            return false;
        }
        return this->_c.isApprox(rhs._c);
    }

    /**
     * \brief Generate a new polynomial by adding 2 polynomials
     */
    polynomial operator+(const polynomial &rhs) const noexcept
    {
        auto sz = std::max(this->_c.size(), rhs._c.size());
        Eigen::ArrayXd ret_c = Eigen::ArrayXd::Zero(sz);
        if (_c.size() == rhs._c.size()) {
            ret_c = _c + rhs._c;
        } else {
            if (this->_c.size() > rhs._c.size()) {
                ret_c.tail(rhs._c.size()) = rhs._c;
                ret_c = ret_c + _c;
            } else {
                ret_c.tail(_c.size()) = _c;
                ret_c = ret_c + rhs._c;
            }
        }

        // Find the index of first element that is non-zero
        Eigen::Index i;
        for (i = 0; i < ret_c.size(); ++i) {
            if (std::abs(ret_c[i])
                > 100 * std::numeric_limits<double>::epsilon()) {
                break;
            }
        }
        return polynomial(ret_c.tail(ret_c.size() - i));
    }

    polynomial operator+(double k) const noexcept
    {
        auto c = _c;
        c[c.size() - 1] += k;
        return polynomial(c);
    }

    friend polynomial operator+(double k, const polynomial &other) noexcept
    {
        auto c = other._c;
        c[c.size() - 1] += k;
        return polynomial(c);
    }

    /**
     * \brief Negation
     */
    polynomial operator-() const noexcept
    {
        return polynomial(-this->_c);
    }

    /**
     * \brief Subtract
     */
    polynomial operator-(const polynomial &rhs) const noexcept
    {
        return -rhs + *this;
    }

    polynomial operator-(double k) const noexcept
    {
        auto c = _c;
        c[c.size() - 1] -= k;
        return polynomial(c);
    }

    friend polynomial operator-(double k, const polynomial &other) noexcept
    {
        Eigen::ArrayXd c = -other._c;
        c[c.size() - 1] += k;
        return polynomial(c);
    }

    /**
     * \brief Production
     */
    polynomial operator*(const polynomial &rhs) const noexcept
    {
        auto n1 = this->_c.size();
        auto n2 = rhs._c.size();
        auto n = n1 + n2 - 1;
        Eigen::ArrayXd cr(n);
        cr.setZero();
        for (Eigen::Index i = 0; i < n; ++i) {
            for (Eigen::Index j = std::max(0ll, i + 1 - n2);
                 j <= std::min(i, n1 - 1); ++j) {
                cr[i] += this->_c[j] * rhs._c[i - j];
            }
        }
        return polynomial(cr);
    }

    /**
     * \brief Production
     */
    polynomial operator*(double k) const noexcept
    {
        return polynomial(k * _c);
    }

    /**
     * \brief Production
     */
    friend polynomial operator*(double k, const polynomial &other) noexcept
    {
        return polynomial(k * other._c);
    }

    /**
     * \brief Production
     */
    polynomial operator/(double k) const
    {
        if (fabs(k) < 100 * std::numeric_limits<double>::epsilon()) {
            throw std::invalid_argument("cannot devide by zero");
        }
        return polynomial(_c / k);
    }

    polynomial operator^(int k) const
    {
        throw std::runtime_error("Not implemented yet");
    }

    /**
     * \brief Return the degree of a polynomial
     */
    auto degree() const noexcept
    {
        return _c.size() - 1;
    }

    /**
     * \brief Return the derivative w.r.t its independent variable of a
     * polynomial
     */
    polynomial derivative() const
    {
        if (_c.size() == 1) return polynomial(Eigen::ArrayXd::Zero(1));

        Eigen::ArrayXd c(_c.size() - 1);
        for (Eigen::Index i = 0; i < c.size(); ++i) {
            c[i] = _c[i] * (_c.size() - 1 - i);
        }
        return polynomial(c);
    }

    /**
     * \brief Returns the coefficients of the polynomial
     */
    const auto &c() const
    {
        return _c;
    }

    /**
     * \brief Returns the roots of the polynomial
     */
    const auto roots() const
    {
        // Make sure the degree is at least 1
        if (degree() < 1) {
            throw std::runtime_error(
                "Degree of the polynomial must be at least 1");
        }
        auto n = degree();
        // Build companion matrix
        Eigen::MatrixXd A(n, n);
        A.setZero();
        // Last col is _c.tail(n)/_c(0) (monic polynomial)
        A.row(0) = -_c.tail(n) / _c(0);
        // Bottom left corner is a identity matrix
        A.bottomLeftCorner(n - 1, n - 1)
             = Eigen::MatrixXd::Identity(n - 1, n - 1);
        return A.eigenvalues();
    }
};
} // namespace details

///////////////////////////////////////////////////////////////////////////////
/// \class tf
/// \brief Transfer function representation
///////////////////////////////////////////////////////////////////////////////
class tf
{
private:
    Eigen::ArrayXd _n;
    Eigen::ArrayXd _d;

public:
    tf() = delete;

    tf(double k)
        : tf((Eigen::Array<double, 1, 1>() << 1).finished(),
             (Eigen::Array<double, 1, 1>() << 1).finished())
    {
    }

    tf(Eigen::ArrayXd n, Eigen::ArrayXd d) : _n(n), _d(d)
    {
        if (_d.size() < 1) {
            throw std::invalid_argument(
                "Denominator must have at least 1 element");
        }
    }

    /**
     * \brief Evaluate the tf at a certain frequency
     */
    std::complex<double> eval(const std::complex<double> &s) const
    {
        details::polynomial den(_d);
        details::polynomial num(_n);
        auto dval = den.eval(s);
        if (std::abs(dval) < 100 * std::numeric_limits<double>::epsilon()) {
            throw std::invalid_argument("Denominator is zero");
        }
        return num.eval(s) / dval;
    }

    friend std::ostream &operator<<(std::ostream &os, const tf &rhs)
    {
        os << details::polynomial(rhs._n) << "\n";
        os << "------------------------------------------------"
           << "\n";
        os << details::polynomial(rhs._d) << "\n";
        return os;
    }

    /**
     * \brief DC gain of the system
     */
    auto dcGain() const
    {
        std::complex<double> zero(0, 0);
        return eval(zero);
    }

    /**
     * \brief Equality test
     */
    bool operator==(const tf &rhs) const
    {
        // a lambda to check if two arrays are equal
        auto isEqual = [](const Eigen::ArrayXd &a1, const Eigen::ArrayXd &a2) {
            return a1.size() != a2.size() || a1.isApprox(a2);
        };
        return isEqual(_n, rhs._n) && isEqual(_d, rhs._d);
    }

    /**
     * \brief Inequality test
     */
    bool operator!=(const tf &rhs) const
    {
        return !(*this == rhs);
    }

    /**
     * \brief Negate a tf
     */
    tf operator-() const
    {
        return tf(-_n, _d);
    }

    /**
     * \brief Connect two systems in parallel
     */
    tf operator+(const tf &rhs) const
    {
        auto num = details::polynomial(_n) * details::polynomial(rhs._d)
                   + details::polynomial(rhs._n) * details::polynomial(_d);
        auto den = details::polynomial(_d) * details::polynomial(rhs._d);
        return tf(num.c(), den.c());
    }

    tf operator+(double k) const
    {
        auto num = details::polynomial(_n) + details::polynomial(k * _d);
        auto den = details::polynomial(_d);
        return tf(num.c(), den.c());
    }

    /**
     * \brief Connect two systems in series
     */
    tf operator*(const tf &rhs) const
    {
        auto num = details::polynomial(_n) * details::polynomial(rhs._n);
        auto den = details::polynomial(_d) * details::polynomial(rhs._d);
        return tf(num.c(), den.c());
    }

    tf operator/(const tf &rhs) const
    {
        auto num = details::polynomial(_n) * details::polynomial(rhs._d);
        auto den = details::polynomial(_d) * details::polynomial(rhs._n);
        return tf(num.c(), den.c());
    }

    /**
     * \brief Gain
     */
    tf operator*(double k) const
    {
        auto num = details::polynomial(k * _n);
        auto den = details::polynomial(_d);
        return tf(num.c(), den.c());
    }

    /**
     * \brief Gain
     */
    friend tf operator*(double k, const tf &other) noexcept
    {
        auto num = details::polynomial(k * other._n);
        auto den = details::polynomial(other._d);
        return tf(num.c(), den.c());
    }
};

///////////////////////////////////////////////////////////////////////////////
/// \class css
/// \brief Contiuous state space representation
///////////////////////////////////////////////////////////////////////////////
using css = details::stateSpace<false>;

///////////////////////////////////////////////////////////////////////////////
/// \class dss
/// \brief Discrete state space representation
///////////////////////////////////////////////////////////////////////////////
using dss = details::stateSpace<true>;

const tf &s = tf((Eigen::Array2d() << 1, 0).finished(),
                 (Eigen::Array<double, 1, 1>() << 1).finished());

template <typename T>
struct is_css {
    static const bool value = false;
};

template <>
struct is_css<css> {
    static const bool value = true;
};

template <typename T>
struct is_dss {
    static const bool value = false;
};

template <>
struct is_dss<dss> {
    static const bool value = true;
};

} // namespace lti
