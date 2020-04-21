#define CATCH_CONFIG_MAIN // This tells Catch to provide a main() - only do this
                          // in one cpp file
#include "catch2/catch.hpp"
#include <iostream>

#include "lti.hpp"

TEST_CASE("Test css", "[test_css]")
{
    SECTION("Test ctor") {
        Eigen::Matrix2d A;
        A.setRandom();
        REQUIRE_NOTHROW(lti::css(A));
    }
}

TEST_CASE("Test polynomial eval", "[test_polynomial]")
{
    using namespace lti::details;
    SECTION("0th order")
    {
        Eigen::VectorXd c(1);
        c << 1;
        auto p = polynomial(c);
        CHECK(p.eval(0) == 1);
        CHECK(p.eval(1) == 1);
    }

    SECTION("1st order")
    {
        Eigen::VectorXd c(2);
        c << 1, 2;
        auto p = polynomial(c);
        REQUIRE(p.eval(0) == 2);
        REQUIRE(p.eval(1) == 3);
        REQUIRE(p.eval(2) == 4);
    }

    SECTION("2nd order")
    {
        Eigen::VectorXd c(3);
        c << 2, 1, 2;
        auto p = polynomial(c);
        REQUIRE(p.eval(0) == 2);
        REQUIRE(p.eval(1) == 5);
        REQUIRE(p.eval(2) == 12);
    }

    // SECTION("Solve linear system")
    // {
    //     Eigen::VectorXd c(3);
    //     c << 1 , 2, 0;
    //     Eigen::MatrixXd A = c.asDiagonal();
    //     Eigen::MatrixXd b = Eigen::MatrixXd::Random(3,3);

    //     Eigen::MatrixXd x = A.fullPivLu().solve(b);

    //     std::cout << "A: \n" << A << "\n";
    //     std::cout << "B: \n" << b << "\n";
    //     std::cout << "x: \n" << x << "\n";

    //     std::cout << "error:\n" << (A*x - b).norm() << "\n";

    //     REQUIRE(false);
    // }
}