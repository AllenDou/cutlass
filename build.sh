echo "-DCUTLASS_NVCC_ARCHS=89 for ampere"
echo "-DCUTLASS_NVCC_ARCHS=90a for hopper"
mkdir -p build && cd build && cmake .. -DCUTLASS_NVCC_ARCHS=89
