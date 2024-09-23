#include <cuda_runtime_api.h>
#include <iostream>
#include <cstdlib>
#include <chrono>

int A[3][4] = {{1,1,1,1},
                   {1,1,1,1},
                   {2,2,2,2}};

int B[4][2] = {{5, 10},               
                   {5, 10},
                   {5, 10},
                   {5, 10}};

int C[3][2] = { 0, 0, 0 ,0, 0, 0};

int vector_dot(const int * v1, const int * v2, int vlen)
{
    int rtn = NULL;
    for(int i = 0; i < vlen; i++)
    {
        rtn += v1[i] * v2[i]; 
    }
    return rtn;
}

template<std::size_t NROWS, std::size_t NCOLS, std::size_t VLEN>
void matMulCPU(int (&mA)[NROWS][VLEN], int (&mB)[VLEN][NCOLS], int (&mC)[NROWS][NCOLS])
{
    for(int i = 0; i < NROWS; i++)
    {
        for(int j = 0; j < NCOLS; j++)
        {
            for(int k = 0; k < VLEN; k++)
            {
                mC[i][j] += mA[i][k] * mB[k][j];
            }
        }
    }
}


int main()
{
    const unsigned int nrows = 30;
    const unsigned int ncols = 20;
    const unsigned int vlen  = 40;

    int _A[nrows][vlen];
    int _B[vlen][ncols];
    int _C[nrows][ncols];

    // Initialize _A
    for (int i = 0; i < nrows; i++)
    {
        for(int j = 0; j < vlen; j++)
        {
            _A[i][j] = std::rand()%10;
        }
    }
    // Initialize _B
    for (int i = 0; i < vlen; i++)
    {
        for(int j = 0; j < ncols; j++)
        {
            _B[i][j] = std::rand()%10;
        }
    }
    // Initialize _C
    for (int i = 0; i < nrows; i++)
    {
        for(int j = 0; j < ncols; j++)
        {
            _C[i][j] = 0;
        }
    }

    // Matrix Multiplication
    auto start = std::chrono::high_resolution_clock::now();
    matMulCPU<nrows,ncols,vlen>(_A, _B, _C);
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
//        std::cout << "C[" << row << "][" << col << "] = " << C[row][col] <<std::endl;
//    }

    // Print small Matrices
    if(nrows * ncols < 20)
    {
        for(int i = 0; i < nrows * vlen; i++){
            const int row = i/vlen;
            const int col = i%vlen;
            std::cout << "A[" << row << "][" << col << "] = " << _A[row][col] <<std::endl;
        }
        for(int i = 0; i < vlen * ncols; i++){
            const int row = i/ncols;
            const int col = i%ncols;
            std::cout << "B[" << row << "][" << col << "] = " << _B[row][col] <<std::endl;
        }
        for(int i = 0; i < nrows * ncols; i++){
            const int row = i/ncols;
            const int col = i%ncols;
            std::cout << "C[" << row << "][" << col << "] = " << _C[row][col] <<std::endl;
        }
    }
    return 0;
}
