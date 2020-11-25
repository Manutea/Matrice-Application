#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "thrust/device_vector.h"
#include "thrust/host_vector.h"

const thrust::host_vector<int> launchAdd(const int _width, const int _height, const int _width2, const int _height2, const thrust::host_vector<int>& h_matrix1, const thrust::host_vector<int>& h_matrix2);
const thrust::host_vector<int> launchMinus(const int _width, const int _height, const int _width2, const int _height2, const thrust::host_vector<int>& h_matrix1, const thrust::host_vector<int>& h_matrix2);
const thrust::host_vector<int> launchHadamard(const int _width, const int _height, const int _width2, const int _height2, const thrust::host_vector<int>& h_matrix1, const thrust::host_vector<int>& h_matrix2);
const thrust::host_vector<int> launchRotation90Dgr(const int _width, const int _height, const thrust::host_vector<int>& h_matrix);
const thrust::host_vector<int> launchUpperTriangular(const int _width, const int _height, const thrust::host_vector<int>& h_matrix);
const std::tuple<int, int, thrust::host_vector<int>> launchMultiplication(const int _width, const int _height, const int _width2, const int _height2, const thrust::host_vector<int>& h_matrix1, const thrust::host_vector<int>& h_matrix2);
const std::tuple<int, int, thrust::host_vector<int>> launchTransposition(const int _width, const int _height, const thrust::host_vector<int>& h_matrix);