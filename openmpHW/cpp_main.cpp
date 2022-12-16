/**
 * Main file for the assignments for Parallel Computing.
 * You should not need to modify this file while working on the assignments.
 * Simply link this file with your implemented parallel routines.
 * 
 * Implemented in 2021 by Emil Loevbak (emil.loevbak@kuleuven.be).
 */

#include <iostream>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/banded.hpp>
#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublas/operation.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <chrono>
#include <random>

namespace ublas = boost::numeric::ublas;

void reference(ublas::matrix<double> &left, ublas::matrix<double> &right, ublas::matrix<double> &result)
{
    result = ublas::prod(left, right);
}
void fullTimesDiagonal(ublas::matrix<double> &left, ublas::matrix<double> &right, ublas::matrix<double> &result);
void fullTimesFull(ublas::matrix<double> &left, ublas::matrix<double> &right, ublas::matrix<double> &result);
void fullTimesFullBlocked(ublas::matrix<double> &left, ublas::matrix<double> &right, ublas::matrix<double> &result);
void triangularTimesFull(ublas::matrix<double> &left, ublas::matrix<double> &right, ublas::matrix<double> &result);

void testMatrixMethod(ublas::matrix<double> left, ublas::matrix<double> right, void (*userProduct)(ublas::matrix<double> &, ublas::matrix<double> &, ublas::matrix<double> &), std::string functionName)
{
    size_t const warmup = 5;
    size_t const iters = 10;
    auto matrixProductBlas = ublas::prod(left, right);
    ublas::matrix<double> matrixProductUser(right);
    for (size_t i = 0; i < warmup; ++i)
    {
        (*userProduct)(left, right, matrixProductUser);
    }
    auto start = std::chrono::system_clock::now();
    for (size_t i = 0; i < iters; ++i)
    {
        (*userProduct)(left, right, matrixProductUser);
    }
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;

    std::cout << "Ran test for " << functionName << ", time taken (ms): " << elapsed_seconds.count() * 1000 / iters << ", error: " << ublas::norm_frobenius(matrixProductBlas - matrixProductUser) / ublas::norm_frobenius(matrixProductBlas) << std::endl;
}

int main(int argc, char *argv[])
{
    if (argc != 2) {
        std::cout << "Expected precisely one command line argument, the matrix dimension!" << std::endl;
        return EXIT_FAILURE;
    }

    size_t N = strtoul(argv[1], NULL, 0);
    std::random_device device;
    std::mt19937 generator(device());
    std::normal_distribution<double> distribution(0.0, 1.0);

    ublas::matrix<double> left(N, N);
    for (size_t i = 0; i < N; ++i)
    {
        for (size_t j = 0; j < N; ++j)
        {
            left(i, j) = distribution(generator);
        }
    }
    //testMatrixMethod(left, left, reference, "reference");

    ublas::diagonal_matrix<double> rightDiagonal(N);
    for (size_t i = 0; i < N; ++i)
    {
        rightDiagonal(i, i) = distribution(generator);
    }
    //testMatrixMethod(left, rightDiagonal, fullTimesDiagonal, "fullTimesDiagonal");

    ublas::matrix<double> rightFull(N, N);
    for (size_t i = 0; i < N; ++i)
    {
        for (size_t j = 0; j < N; ++j)
        {
            rightFull(i, j) = distribution(generator);
        }
    }
    testMatrixMethod(left, rightFull, fullTimesFull, "fullTimesFull");
    //testMatrixMethod(left, rightFull, fullTimesFullBlocked, "fullTimesFullBlocked");

    ublas::triangular_matrix<double, ublas::upper> leftTriangular(N, N);
    for (size_t i = 0; i < N; ++i)
    {
        for (size_t j = 0; j < N; ++j)
        {
            if (i <= j)
            {
                leftTriangular(i, j) = distribution(generator);
            }
        }
    }
    //testMatrixMethod(leftTriangular, rightFull, triangularTimesFull, "triangularTimesFull");
}
