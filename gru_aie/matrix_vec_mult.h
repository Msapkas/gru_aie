#ifndef matrix_vec_mult_h
#define matrix_vec_mult_h

#include <adf.h>

template<int vector_size, int h_vector_size>
void matrix_vec_mult(v8float* matrix,
					 v8float* vector,
				 	 v8float* result);


template<int vector_size, int h_vector_size>
void matrix_vec_mult(v8float* matrix,
					 v8float* vector,
					 v8float* result) {

	for (int i = 0; i < h_vector_size; i++)
		result[i] = null_v8float();

	// Cycle over input vectors
	for (int inp_vec = 0; inp_vec < vector_size; inp_vec++)
		// Cycle over input vector elements
		for (int inp_elem = 0; inp_elem < 8; inp_elem++)
			// Cycle over output buffer
			for (int out_vec = 0; out_vec < h_vector_size; out_vec++)
							result[out_vec] = fpmac(
											result[out_vec],
											vector[inp_vec],
											inp_elem,
											0x0,
											matrix[out_vec+inp_elem*h_vector_size+inp_vec*h_vector_size*8],
											0x0,
											0x76543210
										);
}

#endif
