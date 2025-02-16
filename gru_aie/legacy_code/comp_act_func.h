#ifndef act_func_h
#define act_func_h

#include <adf.h>

template<int vector_size>
void act_func(v8float* input, v8float* output, float max_range, float c0, float c1, float c2);

template<int vector_size>
void act_func(v8float* input, v8float* output, float max_range, float c0, float c1, float c2) {

	v8float pos_range;
	v8float neg_range;
	v8float coeff_0;
	v8float coeff_1;
	v8float coeff_2;

	for(int i=0; i<8; i++){
//		chess_prepare_for_pipelining
//		chess_flatten_loop
		pos_range = upd_elem(pos_range, i, max_range);
		neg_range = upd_elem(neg_range, i,-max_range);
		coeff_0 = upd_elem(coeff_0, i, c0);
		coeff_1 = upd_elem(coeff_1, i, c1);
		coeff_2 = upd_elem(coeff_2, i, c2);
	}

	v8float x_3;
	v8float input_3;

	for(int i=0; i<vector_size; i++){
		input[i] = fpmin(input[i],
						pos_range);

		input[i] = fpmax(input[i],
						neg_range);

		input_3 = fpmul(input[i],
					input[i]);

		input_3 = fpmul(input_3,
					input[i]);

		output[i] = fpmac(coeff_0,
						  input[i],
						  coeff_1);

		output[i] = fpmac(output[i],
						  input_3,
						  coeff_2);
	};
};

#endif
