#ifndef CONFIG_H
#define CONFIG_H

constexpr unsigned int VECTOR_LANES = 4;
constexpr unsigned int X_VECTOR_SIZE = 31;
constexpr unsigned int H_VECTOR_SIZE = 31;
constexpr unsigned int DIST_COEFF = 1;

constexpr unsigned int NKERNELS = H_VECTOR_SIZE/DIST_COEFF;

constexpr unsigned int LUT_SIZE = 4096; // This is half a Tiles Memory (16KB)

#endif