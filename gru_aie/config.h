#ifndef CONFIG_H
#define CONFIG_H

constexpr int VECTOR_LANES = 4;
constexpr int X_VECTOR_SIZE = 32;
constexpr int H_VECTOR_SIZE = 32;
constexpr int DIST_COEFF = 1;

constexpr int NKERNELS = H_VECTOR_SIZE/DIST_COEFF;

constexpr int LUT_SIZE = 4096; // This is half a Tiles Memory (16KB)

#endif