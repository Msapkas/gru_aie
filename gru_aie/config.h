#ifndef CONFIG_H
#define CONFIG_H

constexpr unsigned int VECTOR_LANES = 4;

constexpr unsigned int X_VECTOR_SIZE = 4;
constexpr unsigned int H_VECTOR_SIZE = 4;

constexpr unsigned int DIST_COEFF = 1;

constexpr float SIGMOID_THR = 6;
constexpr float TANH_THR = 3;

constexpr unsigned int LUT_SIZE = 4096;

constexpr unsigned int NKERNELS = H_VECTOR_SIZE/DIST_COEFF;

#endif
