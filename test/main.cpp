#define CATCH_CONFIG_MAIN // This tells Catch to provide a main() - only do this
                          // in one cpp file
#include "catch2/catch.hpp"
#include <iostream>

#include "lti.hpp"

TEST_CASE("Test continuous state space")
{
    int nx = 2, nu = 2, ny = 2;
    Eigen::MatrixXd A(nx, nx);
    Eigen::MatrixXd B(nx, nu);
    Eigen::MatrixXd C(ny, nx);
    Eigen::MatrixXd D(ny, nu);

    SECTION("Ctor")
    {
        REQUIRE_NOTHROW(lti::css(A));
        REQUIRE_NOTHROW(lti::css(A, B));
        REQUIRE_NOTHROW(lti::css(A, B, C));
        REQUIRE_NOTHROW(lti::css(A, B, C, D));

        Eigen::MatrixXd A_bad(nx, nx + 1);
        Eigen::MatrixXd B_bad(nx + 1, nu);
        Eigen::MatrixXd C_bad(ny, nx + 1);
        Eigen::MatrixXd D_bad(ny + 1, nu + 1);

        REQUIRE_THROWS(lti::css(A_bad));
        REQUIRE_THROWS(lti::css(A, B_bad));
        REQUIRE_THROWS(lti::css(A, B, C_bad));
        REQUIRE_THROWS(lti::css(A, B, C_bad));
        REQUIRE_THROWS(lti::css(A, B, C, D_bad));
    }

    SECTION("Operator +")
    {
        int nx = 2, nu = 2, ny = 2;
        Eigen::MatrixXd A(nx, nx);
        Eigen::MatrixXd B(nx, nu);
        Eigen::MatrixXd C(ny, nx);
        Eigen::MatrixXd D(ny, nu);

        A.setIdentity();
        B.setIdentity();
        C.setIdentity();
        D.setZero();
        auto sys = lti::css(A, B, C, D);

        Eigen::MatrixXd A_e(2 * nx, 2 * nx);
        Eigen::MatrixXd B_e(2 * nx, nu);
        Eigen::MatrixXd C_e(ny, 2 * nx);
        Eigen::MatrixXd D_e(ny, nu);

        A_e.setIdentity();
        B_e << 1, 0, 0, 1, 1, 0, 0, 1;
        C_e << 1, 0, 1, 0, 0, 1, 0, 1;
        D_e.setZero();
        auto sys_e = lti::css(A_e, B_e, C_e, D_e);
        CHECK(sys + sys == sys_e);
    }

    SECTION("Negation")
    {
        A.setIdentity();
        B.setIdentity();
        C.setIdentity();
        D.setZero();
        auto sys = lti::css(A, B, C, D);
        auto sys_e = lti::css(A, B, -C, -D);
        CHECK(-sys == sys_e);
    }

    SECTION("Operator -")
    {
        int nx = 2, nu = 2, ny = 2;
        Eigen::MatrixXd A(nx, nx);
        Eigen::MatrixXd B(nx, nu);
        Eigen::MatrixXd C(ny, nx);
        Eigen::MatrixXd D(ny, nu);
        A.setIdentity();
        B.setIdentity();
        C.setIdentity();
        D.setZero();
        auto sys = lti::css(A, B, C, D);

        Eigen::MatrixXd A_e(2 * nx, 2 * nx);
        Eigen::MatrixXd B_e(2 * nx, nu);
        Eigen::MatrixXd C_e(ny, 2 * nx);
        Eigen::MatrixXd D_e(ny, nu);

        A_e.setIdentity();
        B_e << 1, 0, 0, 1, 1, 0, 0, 1;
        C_e << -1, 0, 1, 0, 0, -1, 0, 1;
        D_e.setZero();
        auto sys_e = lti::css(A_e, B_e, C_e, D_e);

        CHECK(sys - sys == sys_e);
    }

    SECTION("Test == operator")
    {
        int nx = 3, nu = 2, ny = 3;
        Eigen::MatrixXd A(nx, nx);
        Eigen::MatrixXd B(nx, nu);
        Eigen::MatrixXd C(ny, nx);
        Eigen::MatrixXd D(ny, nu);

        A.setRandom();
        auto ss = lti::css(A);

        Eigen::MatrixXd A1 = A;
        auto ss1 = lti::css(A1);
        REQUIRE(ss == ss1);

        Eigen::MatrixXd A2 = A;
        A2.setRandom();
        auto ss2 = lti::css(A2);
        REQUIRE_FALSE(ss == ss2);
    }

    SECTION("Test != operator")
    {
        Eigen::Matrix2d A1;
        Eigen::Matrix2d A2;
        A1.setRandom();
        A2.setOnes();

        auto s1 = lti::css(A1);
        auto s2 = lti::css(A2);

        REQUIRE(s1 != s2);
        REQUIRE_FALSE(s1 != s1);
    }
}

TEST_CASE("Test tf")
{
    using namespace std::complex_literals;
    Eigen::VectorXd num(1);
    Eigen::VectorXd den(2);
    num << 1;
    den << 2, 1;
    REQUIRE_NOTHROW(lti::tf(num, den));
    lti::tf sys(num, den);

    SECTION("ctor")
    {
        Eigen::Vector2d num;
        Eigen::VectorXd den;
        num.setRandom();
        // No element in den should throw
        REQUIRE_THROWS(lti::tf(num, den));
    }

    SECTION("Evaluate")
    {
        CHECK(sys.eval(1.0 + 0.0i) == 1.0 / (3.0 + 0.0i));
        CHECK(sys.eval(0.0 + 1.0i) == 1.0 / (1.0 + 2.0i));
        CHECK_THROWS(sys.eval(-0.5 + 0.0i));
    }

    SECTION("dcGain")
    {
        CHECK(sys.dcGain() == 1.0 + 0.0i);
        CHECK(sys.dcGain() == sys.eval(0.0 + 0.0i));
    }

    SECTION("Negation")
    {
        lti::tf sys_e(-num, den);
        CHECK(-sys == sys_e);
    }

    SECTION("Operator + tf")
    {
        Eigen::ArrayXd num_e(2);
        Eigen::ArrayXd den_e(3);
        num_e << 4, 2;
        den_e << 4, 4, 1;
        lti::tf sys_e(num_e, den_e);
        CHECK(sys + sys == sys_e);
    }

    SECTION("Operator *")
    {
        Eigen::ArrayXd num_e(2);
        Eigen::ArrayXd den_e(3);
        num_e << 4, 2;
        den_e << 4, 4, 1;
        lti::tf sys_e(num_e, den_e);
        CHECK(sys + sys == sys_e);
    }
}

TEST_CASE("Test polynomial")
{
    using namespace lti::details;
    using namespace std::complex_literals;

    SECTION("Ctor")
    {
        CHECK_NOTHROW(polynomial(Eigen::VectorXd::Random(5)));
        CHECK_NOTHROW(polynomial(Eigen::VectorXd::Zero(1)));
        CHECK_THROWS(polynomial(Eigen::VectorXd::Zero(2)));
    }

    SECTION("Evaluate 0th order poly")
    {
        Eigen::VectorXd c(1);
        c << 1;
        auto p = polynomial(c);
        CHECK(p.eval(0) == 1);
        CHECK(p.eval(1.0i) == 1.0 + 0.0i);
        CHECK(p.eval(10) == 1);
        CHECK(p.eval(1.0 + 1.0i) == 1.0 + 0.0i);
    }

    SECTION("Evaluate 1st order poly")
    {
        Eigen::VectorXd c(2);
        c << 1, 2;
        auto p = polynomial(c);
        REQUIRE(p.eval(0) == 2);
        REQUIRE(p.eval(1) == 3);
        REQUIRE(p.eval(2) == 4);

        REQUIRE(p.eval(0.0i) == 2.0 + 0.0i);
        REQUIRE(p.eval(1.0i) == 2.0 + 1.0i);
        REQUIRE(p.eval(2.0i) == 2.0 + 2.0i);
    }

    SECTION("Evaluate 2nd order poly")
    {
        Eigen::VectorXd c(3);
        c << 2, 1, 2;
        auto p = polynomial(c);
        REQUIRE(p.eval(0) == 2);
        REQUIRE(p.eval(1) == 5);
        REQUIRE(p.eval(2) == 12);
    }

    SECTION("Equality test")
    {
        auto p = polynomial(Eigen::VectorXd::Random(3));
        auto p_ones = polynomial(Eigen::VectorXd::Ones(3));
        auto p_lo = polynomial(Eigen::VectorXd::Ones(2));
        CHECK(p == p);
        CHECK_FALSE(p == p_ones);
        CHECK_FALSE(p == p_lo);
    }

    SECTION("Add +-* with a double")
    {
        Eigen::ArrayXd c(2);
        c << 1, 2;
        auto p = polynomial(c);

        // + test
        Eigen::ArrayXd c_add_e(2);
        c_add_e << 1, 3;
        auto p_add = polynomial(c_add_e);
        CHECK(p + 1 == p_add);
        CHECK(1 + p == p_add);

        // - test
        Eigen::ArrayXd c_minus_r_e(2); // 1 - poly
        Eigen::ArrayXd c_minus_l_e(2); // ploy - 1
        c_minus_r_e << -1, -1;
        c_minus_l_e << 1, 1;
        auto p_minus_r = polynomial(c_minus_r_e);
        auto p_minus_l = polynomial(c_minus_l_e);
        CHECK(1 - p == p_minus_r);
        CHECK(p - 1 == p_minus_l);

        // * test
        Eigen::ArrayXd c_mul_e(2);
        c_mul_e << 2, 4;
        auto p_mul = polynomial(c_mul_e);
        CHECK(p * 2 == p_mul);
        CHECK(2 * p == p_mul);
    }

    SECTION("Add two polynomial with the same degree")
    {
        Eigen::VectorXd c(2);
        c << 1, 2;
        auto p = polynomial(c);
        auto pp = p + p;

        Eigen::VectorXd c_expect(2);
        c_expect << 2, 4;
        auto p_expect = polynomial(c_expect);
        CHECK(pp == p_expect);
    }

    SECTION("Add two polynomial with different degrees")
    {
        Eigen::VectorXd c1(2);
        Eigen::VectorXd c2(3);
        c1 << 1, 2;
        c2 << 1, 2, 3;
        auto p1 = polynomial(c1);
        auto p2 = polynomial(c2);

        Eigen::VectorXd c_expect(3);
        c_expect << 1, 3, 5;
        auto p_expect = polynomial(c_expect);
        CHECK(p1 + p2 == p_expect);
    }

    SECTION("Add two polynomial that reduces 1 degree")
    {
        Eigen::VectorXd c1(3);
        Eigen::VectorXd c2(3);
        c1 << 1, 2, 3;
        c2 << -1, 2, 3;
        auto p1 = polynomial(c1);
        auto p2 = polynomial(c2);
        auto p = p1 + p2;

        Eigen::VectorXd c_expect(2);
        c_expect << 4, 6;
        auto p_expect = polynomial(c_expect);
        CHECK(p == p_expect);
    }

    SECTION("Add two polynomial that reduces 2 degrees")
    {
        Eigen::VectorXd c1(3);
        Eigen::VectorXd c2(3);
        c1 << 1, 2, 3;
        c2 << -1, -2, 3;
        auto p1 = polynomial(c1);
        auto p2 = polynomial(c2);
        auto p = p1 + p2;

        Eigen::VectorXd c_expect(1);
        c_expect << 6;
        auto p_expect = polynomial(c_expect);
        CHECK(p == p_expect);
    }

    SECTION("Negate")
    {
        Eigen::VectorXd c(2);
        c << 1, 2;
        auto p = polynomial(c);

        Eigen::VectorXd c_expect(2);
        c_expect << -1, -2;
        auto p_expect = polynomial(c_expect);
        CHECK(-p == p_expect);
    }

    SECTION("Subtraction")
    {
        Eigen::VectorXd c1(2);
        Eigen::VectorXd c2(2);
        Eigen::VectorXd c_expect(2);
        c1 << 1, 2;
        c2 << 2, 5;
        c_expect << 1, 3;
        auto p1 = polynomial(c1);
        auto p2 = polynomial(c2);
        auto p = p2 - p1;
        auto p_expect = polynomial(c_expect);
        CHECK(p == p_expect);
    }

    SECTION("Production")
    {
        Eigen::VectorXd c(3);
        Eigen::VectorXd c_expect(5);
        c << 1, 2, 3;
        c_expect << 1, 4, 10, 12, 9;
        auto p = polynomial(c);
        auto p_expect = polynomial(c_expect);
        CHECK(p * p == p_expect);
    }

    SECTION("Degree")
    {
        Eigen::VectorXd c(3);
        c << 1, 2, 3;
        auto p = polynomial(c);
        CHECK(2ll == p.degree());
    }

    SECTION("Derivative")
    {
        Eigen::ArrayXd c(3);
        c << 1, 2, 3;
        auto p = polynomial(c);
        // Check first order derivative
        Eigen::ArrayXd c_expect_1st(2);
        c_expect_1st << 2, 2;
        auto p_expect_1st = polynomial(c_expect_1st);
        CHECK(p.derivative() == c_expect_1st);
        // Check second order derivative
        Eigen::ArrayXd c_expect_2nd(1);
        c_expect_2nd << 2;
        auto p_expect_2nd = polynomial(c_expect_2nd);
        CHECK(p.derivative().derivative() == p_expect_2nd);
        // Check third order derivative
        Eigen::ArrayXd c_expect_3rd(1);
        c_expect_3rd << 0;
        auto p_expect_3rd = polynomial(c_expect_3rd);
        CHECK(p.derivative().derivative().derivative() == p_expect_3rd);
    }
}

TEST_CASE("MAIN")
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