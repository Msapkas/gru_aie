#ifndef CONFIG_H
#define CONFIG_H

// This configuration file should be used to control the parameters of the model
// The vector we want to use. For floats can be 4 or 8
constexpr int VECTOR_LANES = 4;

// The size of the Input Vector. This affects only the amount of macs that will be performed in a row
// and does not affect the amount of kernels used. Must be a multiple of VECTOR LANES.
constexpr int X_VECTOR_SIZE = 8;
// The size of the Hidden State Vector. Must be a multiple of VECTOR LANES.

constexpr int H_VECTOR_SIZE = 20;
// The DISTRIBUTION COEFFICIENT control how many rows a kernel will MAC. So how many elements of the output vector will be
// computed inside a kernel. Therefore also controls how many kernels will be instantiated. This is were parallelization
// is being done.
constexpr int DIST_COEFF = 2;

// As stated above the NUMBER OF KERNELS that will be instatiated is:
constexpr int NKERNELS = H_VECTOR_SIZE/DIST_COEFF;

// 
constexpr int LUT_SIZE = 4096; // This is half a Tiles Memory (16KB)

#endif