#include "matrix.cuh"
#include <iostream>
#include <cstdlib>
#include <ctime>


template <typename T>
void showMatrix(const int _width, const int _height, const T* _m) 
{
    auto cmp = 0;
    for (auto i = 0; i < _width * _height; ++i) {
        std::cout << _m[i] << '\t';
        cmp++;
        if (cmp >= _width) {
            std::cout << '\n';
            cmp = 0;
        }
    }
}

thrust::host_vector<int> fillMatrix(const int _size)
{
    thrust::host_vector<int> matrix(_size);
    for (auto i = 0; i < _size; ++i) {
        matrix[i] = rand() % 8 + 1;
    }
    return matrix;
}

int main()
{
    const auto width = 4;
    const auto height = 5;
    const auto matrixSize = width * height;

    const auto width2 = 5;
    const auto height2 = 5;
    const auto matrixSize2 = width2 * height2;

    srand(static_cast <unsigned int> (time(0)));
    thrust::host_vector<int> h_matrix1 = fillMatrix(matrixSize);
    thrust::host_vector<int> h_matrix2 = fillMatrix(matrixSize2);

    std::cout << "Matrix 1\n";
    showMatrix(width, height, h_matrix1.data());
    std::cout << '\n';

    std::cout << "Matrix 2\n";
    showMatrix(width2, height2, h_matrix2.data());
    std::cout << '\n';

    try
    {
        thrust::host_vector<int> h_result = launchAdd(width, height, width2, height2, h_matrix1, h_matrix2);

        showMatrix(width, height, h_result.data());
        std::cout << '\n';
    }
    catch (std::exception const& exception)
    {
        std::cout << "Une erreur est survenue : " << exception.what() << std::endl;
    }


    return 0;
}

