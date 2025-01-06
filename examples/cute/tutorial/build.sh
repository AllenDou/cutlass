nvcc -o a.out zixiao_test_mma.cu -O2 -arch=sm_89 -std=c++17 -I/root/cutlass_allendou/include/ --expt-relaxed-constexpr -cudart shared --cudadevrt none
