/**
 * Serial implementation of the matrix operations for the assignments for Parallel Computing.
 * You can use a copy of this file as a starting point for each of your parallel implementations.
 * 
 * Implemented in 2021 by Emil Loevbak (emil.loevbak@kuleuven.be).
 */

#include <boost/numeric/ublas/matrix.hpp>

namespace ublas = boost::numeric::ublas;

void fullTimesDiagonal(ublas::matrix<double> &left, ublas::matrix<double> &right, ublas::matrix<double> &result)
{
    size_t N = result.size1();
    int numThreads = 4;
    #pragma omp parallel shared(left, right, result, N) num_threads(numThreads)
    {
        #pragma omp for
        for (size_t i = 0; i < N; ++i)
        {
            for (size_t j = 0; j < N; ++j)
            {
                result(i, j) = left(i, j) * right(j, j);
            }
        }
    }
}

// Lots of false sharing if j-loop if parallized --> slow!
void fullTimesFull(ublas::matrix<double> &left, ublas::matrix<double> &right, ublas::matrix<double> &result)
{
    size_t N = result.size1();
    int numThreads = 4; 
    auto sum = 0.0;
    #pragma omp parallel shared(left, right, result, N) num_threads(numThreads)
    {
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                // result(i, j) = 0.0;
                #pragma omp barrier
                sum = 0.0;
                // #pragma omp parallel for shared(left, right, result, N) num_threads(numThreads) reduction(+: sum)
                #pragma omp for reduction(+: sum)
                for (size_t k = 0; k < N; ++k) {
                    sum += left(i, k) * right(k, j);
                }
                result(i, j) = sum;
            }
        }
    }
}

void fullTimesFullBlocked(ublas::matrix<double> &left, ublas::matrix<double> &right, ublas::matrix<double> &result)
{
    size_t const blocksize = 50;
    size_t N = result.size1();
    int numThreads = 4;
    #pragma omp parallel shared(left, right, result, N) num_threads(numThreads)
    {
        #pragma omp for
        for (size_t i = 0; i < N / blocksize; ++i)
        {
            for (size_t j = 0; j < N / blocksize; ++j)
            {
                for (size_t i_block = 0; i_block < blocksize; ++i_block)
                {
                    for (size_t j_block = 0; j_block < blocksize; ++j_block)
                    {
                        result(i * blocksize + i_block, j * blocksize + j_block) = 0.0;
                    }
                }
                for (size_t k = 0; k < N / blocksize; ++k)
                {
                    for (size_t i_block = 0; i_block < blocksize; ++i_block)
                    {
                        for (size_t j_block = 0; j_block < blocksize; ++j_block)
                        {
                            for (size_t k_block = 0; k_block < blocksize; ++k_block)
                            {
                                result(i * blocksize + i_block, j * blocksize + j_block) += left(i * blocksize + i_block, k * blocksize + k_block) * right(k * blocksize + k_block, j * blocksize + j_block);
                            }
                        }
                    }
                }
            }
        }
    }

}

void triangularTimesFull(ublas::matrix<double> &left, ublas::matrix<double> &right, ublas::matrix<double> &result)
{
    int numThreads = 4;
    #pragma omp parallel shared(left, right, result) num_threads(numThreads)
    {
        size_t N = result.size1();
        #pragma omp for
        for (size_t i = 0; i < N; ++i)
        {
            for (size_t j = 0; j < N; ++j)
            {
                result(i, j) = 0.0;
                for (size_t k = i; k < N; ++k)
                {
                    result(i, j) += left(i, k) * right(k, j);
                }
            }
        }
    }
}