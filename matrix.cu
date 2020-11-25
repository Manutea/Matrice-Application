#include "matrix.cuh"

#include <stdexcept>

#define THREAD_X 8
#define THREAD_Y 8


/**
* @brief The cuda kernel to add two 2D matrices.
*
* @tparam T, the type of value to retrieve.
*
* @param[in] width, the width of the two matrices.
* @param[in] height, the height of the two matrices.
* @param[in] m, the first matrix.
* @param[in] n, the second matrix.
* @param[out] output, 1D array containing the add result.
*
*/
template <typename T>
__global__ void addKernel(const int _width, const int _height, const T* _m, const T* _n, T* _output)
{
    const auto col = threadIdx.x + blockIdx.x * blockDim.x;
    const auto row = threadIdx.y + blockIdx.y * blockDim.y;
    if (col >= _width || row >= _height) return;

    const auto index = row * _width + col;

    _output[index] = _m[index] + _n[index];
}

/**
* @brief Launch the cuda kernel to add two 2D matrices.
*
* @param[in] width, the width of the two matrices.
* @param[in] height, the height of the two matrices.
* @param[in] h_matrix1, the first matrix.
* @param[in] h_matrix2, the second matrix.
*
* @returns the result of the add inside 1D host vector.
*/
const thrust::host_vector<int> launchAdd(const int _width, const int _height, const int _width2, const int _height2, const thrust::host_vector<int>& h_matrix1, const thrust::host_vector<int>& h_matrix2)
{
    if (_width != _width2 || _height != _height2) {
        throw std::length_error("matrix1 width height != matrix2 width height");
    }
    else {
        const dim3 blocks(_width / THREAD_X + 1, _height / THREAD_Y + 1);
        const dim3 threads(THREAD_X, THREAD_Y);

        const thrust::device_vector<int> d_matrix1 = h_matrix1;
        const int* matrix1BufferArray = thrust::raw_pointer_cast(&d_matrix1[0]);

        const thrust::device_vector<int> d_matrix2 = h_matrix2;
        const int* matrix2BufferArray = thrust::raw_pointer_cast(&d_matrix2[0]);

        thrust::device_vector<int> d_result(_width * _height);
        int* resultBufferArray = thrust::raw_pointer_cast(&d_result[0]);

        addKernel << <blocks, threads >> > (_width, _height, matrix1BufferArray, matrix2BufferArray, resultBufferArray);
        cudaDeviceSynchronize();

        const thrust::host_vector<int> h_result = d_result;

        return h_result;
    }
} 


/**
* @brief The cuda kernel to minus two 2D matrices.
*
* @tparam T, the type of value to retrieve.
*
* @param[in] width, the width of the two matrices.
* @param[in] height, the height of the two matrices.
* @param[in] m, the first matrix.
* @param[in] n, the second matrix.
* @param[out] output, 1D array containing the minus result.
*
*/
template <typename T>
__global__ void minusKernel(const int _width, const int _height, const T* _m, const T* _n, T* _output)
{
    const auto col = threadIdx.x + blockIdx.x * blockDim.x;
    const auto row = threadIdx.y + blockIdx.y * blockDim.y;
    if (col >= _width || row >= _height) return;

    const auto index = row * _width + col;

    _output[index] = _m[index] - _n[index];
}

/**
* @brief Launch the cuda kernel to minus two 2D matrices.
*
* @param[in] width, the width of the two matrices.
* @param[in] height, the height of the two matrices.
* @param[in] h_matrix1, the first matrix.
* @param[in] h_matrix2, the second matrix.
*
* @returns the result of the add inside 1D host vector.
*/
const thrust::host_vector<int> launchMinus(const int _width, const int _height, const int _width2, const int _height2, const thrust::host_vector<int>& h_matrix1, const thrust::host_vector<int>& h_matrix2)
{
    if (_width != _width2 || _height != _height2) {
        throw std::length_error("matrix1 width height != matrix2 width height");
    }
    else {
        const dim3 blocks(_width / THREAD_X + 1, _height / THREAD_Y + 1);
        const dim3 threads(THREAD_X, THREAD_Y);

        const thrust::device_vector<int> d_matrix1 = h_matrix1;
        const int* matrix1BufferArray = thrust::raw_pointer_cast(&d_matrix1[0]);

        const thrust::device_vector<int> d_matrix2 = h_matrix2;
        const int* matrix2BufferArray = thrust::raw_pointer_cast(&d_matrix2[0]);

        thrust::device_vector<int> d_result(_width * _height);
        int* resultBufferArray = thrust::raw_pointer_cast(&d_result[0]);

        minusKernel << <blocks, threads >> > (_width, _height, matrix1BufferArray, matrix2BufferArray, resultBufferArray);
        cudaDeviceSynchronize();

        const thrust::host_vector<int> h_result = d_result;
        return h_result;
    }
}


/**
* @brief The cuda kernel to apply Hadamard operation to two 2D matrices.
*
* @tparam T, the type of value to retrieve.
*
* @param[in] width, the width of the two matrices.
* @param[in] height, the height of the two matrices.
* @param[in] m, the first matrix.
* @param[in] n, the second matrix.
* @param[out] output, 1D array containing the Hadamard result.
*
*/
template <typename T>
__global__ void hadamardKernel(const int _width, const int _height, const T* _m, const T* _n, T* _output)
{
    const auto col = threadIdx.x + blockIdx.x * blockDim.x;
    const auto row = threadIdx.y + blockIdx.y * blockDim.y;
    if (col >= _width || row >= _height) return;

    const auto index = row * _width + col;

    _output[index] = _m[index] * _n[index];
}

/**
* @brief Launch the cuda kernel to apply the Hadamard product to two 2D matrices.
*
* @param[in] width, the width of the two matrices.
* @param[in] height, the height of the two matrices.
* @param[in] h_matrix1, the first matrix.
* @param[in] h_matrix2, the second matrix.
*
* @returns the result of the Hadamard product inside 1D host vector.
*/
const thrust::host_vector<int> launchHadamard(const int _width, const int _height, const int _width2, const int _height2, const thrust::host_vector<int>& h_matrix1, const thrust::host_vector<int>& h_matrix2)
{ 
    if (_width != _width2 || _height != _height2) {
        throw std::length_error("matrix1 width height != matrix2 width height");
    }
    else {
        const dim3 blocks(_width / THREAD_X + 1, _height / THREAD_Y + 1);
        const dim3 threads(THREAD_X, THREAD_Y);

        const thrust::device_vector<int> d_matrix1 = h_matrix1;
        const int* matrix1BufferArray = thrust::raw_pointer_cast(&d_matrix1[0]);

        const thrust::device_vector<int> d_matrix2 = h_matrix2;
        const int* matrix2BufferArray = thrust::raw_pointer_cast(&d_matrix2[0]);

        thrust::device_vector<int> d_result(_width * _height);
        int* resultBufferArray = thrust::raw_pointer_cast(&d_result[0]);

        minusKernel << <blocks, threads >> > (_width, _height, matrix1BufferArray, matrix2BufferArray, resultBufferArray);
        cudaDeviceSynchronize();

        const thrust::host_vector<int> h_result = d_result;

        return h_result;
    }
}


/**
* @brief The cuda kernel to apply the multiplication of two matrices.
*
* @tparam T, the type of value to retrieve.
*
* @param[in] width, the width of the m matrix.
* @param[in] height, the height of the n matrix.
* @param[in] n, the first matrix.
* @param[in] m, the second matrix.
* @param[out] output, 1D array containing the multiplication result.
*
*/
template <typename T>
__global__ void multKernel(const int _width, const int _height, const T* _m, const T* _n, T* _output)
{
    const auto col = threadIdx.x + blockIdx.x * blockDim.x;
    const auto row = threadIdx.y + blockIdx.y * blockDim.y;
    if (col >= _width || row >= _height) return;

    T tmpSum = 0;
    for (auto i = 0; i < _width; ++i) {
        tmpSum += _m[row * _width + i] * _n[i * _width + col];
    }
    _output[row * _width + col] = tmpSum;
}

/**
* @brief Launch the cuda kernel to apply the multiplication of two matrices.
*
* @param[in] width, the width of the h_matrix1.
* @param[in] height, the height of the h_matrix1.
* @param[in] width2, the width of the h_matrix2.
* @param[in] height2, the height of the h_matrix2.
* @param[in] h_matrix1, the first matrix.
* @param[in] h_matrix2, the second matrix.
*
* @returns width, height and the result of the Hadamard product inside 1D host vector.
*/
const std::tuple<int, int, thrust::host_vector<int>> launchMultiplication(const int _width, const int _height, const int _width2, const int _height2, const thrust::host_vector<int>& h_matrix1, const thrust::host_vector<int>& h_matrix2)
{
    if (_height != _width2) {
        throw std::length_error("matrix1 height != matrix2 width");
    }
    else {
        const dim3 blocks(_width2 / THREAD_X + 1, _height / THREAD_Y + 1);
        const dim3 threads(THREAD_X, THREAD_Y);

        const thrust::device_vector<int> d_matrix1 = h_matrix1;
        const int* matrix1BufferArray = thrust::raw_pointer_cast(&d_matrix1[0]);

        const thrust::device_vector<int> d_matrix2 = h_matrix2;
        const int* matrix2BufferArray = thrust::raw_pointer_cast(&d_matrix2[0]);

        thrust::device_vector<int> d_result(_width2 * _height);
        int* resultBufferArray = thrust::raw_pointer_cast(&d_result[0]);

        minusKernel << <blocks, threads >> > (_width2, _height, matrix1BufferArray, matrix2BufferArray, resultBufferArray);
        cudaDeviceSynchronize();

        const thrust::host_vector<int> h_result = d_result;

        return std::make_tuple(_width2, _height, h_result);
    }
}


/**
* @brief The cuda kernel to transpose the matrix.
*
* @tparam T, the type of value to retrieve.
*
* @param[in] width, the width of the m matrix.
* @param[in] height, the height of the m matrix.
* @param[in] m, the matrix.
* @param[out] output, 1D array containing the result.
*
*/
template <typename T>
__global__ void transposeKernel(const int _width, const int _height, const T* _m, T* _output)
{
    const auto col = threadIdx.x + blockIdx.x * blockDim.x;
    const auto row = threadIdx.y + blockIdx.y * blockDim.y;
    if (col >= _width || row >= _height) return;

    _output[row + _height * col] = _m[col + _width * row];
}

/**
* @brief Launch the cuda kernel to transpose the matrix.
*
* @param[in] width, the width of the m matrix.
* @param[in] height, the height of the m matrix.
* @param[in] m, the matrix.
*
* @returns the result of the transposed matrix inside a 1D host vector.
*/
const std::tuple<int, int, thrust::host_vector<int>> launchTransposition(const int _width, const int _height, const thrust::host_vector<int>& h_matrix)
{
    const dim3 blocks(_width / THREAD_X + 1, _height / THREAD_Y + 1);
    const dim3 threads(THREAD_X, THREAD_Y);

    const thrust::device_vector<int> d_matrix1 = h_matrix;
    const int* matrixBufferArray = thrust::raw_pointer_cast(&d_matrix1[0]);

    thrust::device_vector<int> d_result(_width * _height);
    int* resultBufferArray = thrust::raw_pointer_cast(&d_result[0]);

    transposeKernel << <blocks, threads >> > (_width, _height, matrixBufferArray, resultBufferArray);
    cudaDeviceSynchronize();

    const thrust::host_vector<int> h_result = d_result;

    return std::make_tuple(_height, _width, h_result);;
}


/**
* @brief The cuda kernel to rotate at 90° the matrix.
*
* @tparam T, the type of value to retrieve.
*
* @param[in] size, the width of the m matrix.
* @param[in] m, the matrix.
* @param[out] output, 1D array containing the result.
*
*/
template <typename T>
__global__ void rotate90MatrixKernel(const int _size, const T* _m, T* _output)
{
    const auto layer = threadIdx.x + blockIdx.x * blockDim.x;
    const auto index = threadIdx.y + blockIdx.y * blockDim.y;

    const auto first = layer;
    const auto last = _size - 1 - layer;

    if (layer >= _size / 2 || index >= last) return;

    const auto offset = index - first;
    //Left -> top
    _output[first * _size + index] = _m[(last - offset) * _size + first];
    //Bottom -> left
    _output[(last - offset) * _size + first] = _m[last * _size + last - offset];
    //Right -> bottom
    _output[last * _size + last - offset] = _m[index * _size + last];
    //Top -> right
    _output[index * _size + last] = _m[first * _size + index];
}

/**
* @brief Launch the cuda kernel to rotate the matrix.
*
* @param[in] width, the width of the m matrix.
* @param[in] height, the height of the m matrix.
* @param[in] m, the matrix.
*
* @returns the result of the rotated matrix inside a 1D host vector.
*/
const thrust::host_vector<int> launchRotation90Dgr(const int _width, const int _height, const thrust::host_vector<int>& h_matrix)
{

    if (_width != _height){
        throw std::length_error(" _width != _height");
    }
    else
    {
        const dim3 blocks(_width / THREAD_X + 1, _height / THREAD_Y + 1);
        const dim3 threads(THREAD_X, THREAD_Y);

        const thrust::device_vector<int> d_matrix1 = h_matrix;
        const int* matrixBufferArray = thrust::raw_pointer_cast(&d_matrix1[0]);

        thrust::device_vector<int> d_result(_width * _height);
        int* resultBufferArray = thrust::raw_pointer_cast(&d_result[0]);

        rotate90MatrixKernel << <blocks, threads >> > (_width, matrixBufferArray, resultBufferArray);
        cudaDeviceSynchronize();

        const thrust::host_vector<int> h_result = d_result;

        return h_result;
    }
}


/**
* @brief The cuda kernel to upper triangular the matrix.
*
* @tparam T, the type of value to retrieve.
*
* @param[in] size, the width of the m matrix.
* @param[out] m, the matrix result.
*
*/
template <typename T>
__global__ void upperTriangularMatrixKernel(const int _size, T* _m)
{
    const auto col = threadIdx.x + blockIdx.x * blockDim.x;
    const auto row = threadIdx.y + blockIdx.y * blockDim.y;

    if (col >= _size - 1 || row <= row + 1 || row >= _size) return;

    T temp = _m[row * _size + col] / _m[col * _size + col];

    for (auto k = 0; k < _size + 1; ++k) {
        _m[row * _size + k] -= (_m[col * _size + k] * temp);
    }

}

/**
* @brief Launch the cuda kernel to upper triangular the matrix.
*
* @param[in] width, the width of the m matrix.
* @param[in] height, the height of the m matrix.
* @param[out] h_matrix, the matrix.
*
* @returns the result of the rotated matrix inside a 1D host vector.
*/
const thrust::host_vector<int> launchUpperTriangular(const int _width, const int _height, const thrust::host_vector<int>& h_matrix)
{
    if (_width != _height){
        throw std::length_error(" _width != _height");
    }
    else {
        const dim3 blocks(_width / THREAD_X + 1, _height / THREAD_Y + 1);
        const dim3 threads(THREAD_X, THREAD_Y);

        thrust::device_vector<int> d_matrix1 = h_matrix;
        int* matrixBufferArray = thrust::raw_pointer_cast(&d_matrix1[0]);

        upperTriangularMatrixKernel << <blocks, threads >> > (_width, matrixBufferArray);
        cudaDeviceSynchronize();

        const thrust::host_vector<int> h_result = d_matrix1;

        return h_result;
    }
}