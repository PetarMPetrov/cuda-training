#include <cuda_runtime_api.h>
#include <iostream>
#include <cstdlib>
#include <chrono>

int A[3*4] = {1,1,1,1, 
              1,1,1,1,
              2,2,2,2};

int B[4*2] = {5, 10,
              5, 10,
              5, 10,
              5, 10};

int C[3*2] = { 0, 0, 0 ,0, 0, 0};

//int vector_dot(const int * v1, const int * v2, int vlen)
//{
//    int rtn = 0;
//    for(int i = 0; i < vlen; i++)
//    {
//        rtn += v1[i] * v2[i]; 
//    }
//    return rtn;
//}

void matMulCPU(int *mA, int *mB, int *mC, std::size_t NROWS, std::size_t NCOLS, std::size_t VLEN)
{
    for(int i = 0; i < NROWS; i++)
    {
        for(int j = 0; j < NCOLS; j++)
        {
            for(int k = 0; k < VLEN; k++)
            {
                mC[i * NCOLS + j] += mA[i * VLEN + k] * mB[k * NCOLS + j];
            }
        }
    }
}

int main(int argc, char *argv[])
{
    unsigned int _nrows, _ncols, _vlen;
    if (argc == 1)
    {
        _nrows = 3000;
        _ncols = 200;
        _vlen  = 4;
    }
    else if (argc == 2)
    {
        _nrows = 3 * atoi(argv[1]);
        _ncols = 2 * atoi(argv[1]);
        _vlen  = 4 * atoi(argv[1]);
    }
    else
    {
        _nrows = atoi(argv[1]);
        _ncols = atoi(argv[2]);
        _vlen  = atoi(argv[3]);
    }

    const unsigned int nrows = _nrows;
    const unsigned int ncols = _ncols;
    const unsigned int vlen = _vlen;

    int _A[nrows * vlen];
    int _B[vlen * ncols];
    int _C[nrows * ncols];

    // Initialize _A
    for (int i = 0; i < nrows * vlen; i++)
    {
        _A[i] = std::rand()%10;
    }
    // Initialize _B
    for (int i = 0; i < vlen * ncols; i++)
    {
        _B[i] = std::rand()%10;
    }
    // Initialize _C
    for (int i = 0; i < nrows * ncols; i++)
    {
        _C[i] = 0;
    }

    // Matrix Multiplication
    auto start = std::chrono::high_resolution_clock::now();
    matMulCPU(_A, _B, _C, nrows, ncols, vlen);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "MatMul finished in " << duration.count() << " microsec" << std::endl;
    //matMulCPU(_A, _B, _C, nrows, ncols, vlen);
/*    for(int i = 0; i < nrows; i++)
    {
        for(int j = 0; j < ncols; j++)
        {
            for(int k = 0; k < vlen; k++)
            {
                _C[i][j] += _A[i][k] * _B[k][j];
            }
        }
    }
*/
//    for(int i = 0; i < 6; i++){
//        const int row = i/2;
//        const int col = i%2;
//        std::cout << "C[" << row << "][" << col << "] = " << C[i] <<std::endl;
//    }

    // Print small Matrices
//    if(nrows * ncols < 20)
//    {
//        for(int i = 0; i < nrows * vlen; i++){
//            const int row = i/vlen;
//            const int col = i%vlen;
//            std::cout << "A[" << row << "][" << col << "] = " << _A[i] <<std::endl;
//        }
//        for(int i = 0; i < vlen * ncols; i++){
//            const int row = i/ncols;
//            const int col = i%ncols;
//            std::cout << "B[" << row << "][" << col << "] = " << _B[i] <<std::endl;
//        }
//        for(int i = 0; i < nrows * ncols; i++){
//            const int row = i/ncols;
//            const int col = i%ncols;
//            std::cout << "C[" << row << "][" << col << "] = " << _C[i] <<std::endl;
//        }
//    }
    return 0;
}
