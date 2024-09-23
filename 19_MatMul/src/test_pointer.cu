#include <iostream>

int A[5] = {11,12,13,14,15};

int main()
{
    int * p = A;
    for(int step = 0; step < 5; step++)
    {
        std::cout << "Element position <" << &A[step] << ">" << std::endl;
        std::cout << "Pointer value <" << (p + step) << ">" << std::endl;
        std::cout << "Reference Element: " << *(p+step) << std::endl;
        std::cout << "Access Pointer: " << p[step] << std::endl;
    }
    return 0;
} 
