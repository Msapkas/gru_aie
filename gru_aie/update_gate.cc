#include "update_gate.h"
#include "matrix_vec_mult.h"
#include "act_func.h"

using namespace adf;

template<int x_input_size, int h_input_size>
void update_gate(
		input_stream<float>* x_input,
		input_stream<float>* h_input,

		output_stream<float>* output,

		const float (&weights_Whu)[h_input_size*h_input_size],
		const float (&weights_Wxu)[x_input_size*h_input_size],
		const float (&biases_u)[h_input_size]

		){

	const int x_vector_size = x_input_size/8;
	const int h_vector_size = h_input_size/8;
	bool tlast = false;

	alignas(32) float x[x_input_size];
	alignas(32) float h[h_input_size];

	v8float* v8_Wxu = (v8float*) &(weights_Wxu[0]);
	v8float* v8_Whu = (v8float*) &(weights_Whu[0]);
	v8float* v8_bu = (v8float*) &(biases_u[0]);

	v8float cmd_buffer = null_v8float();

	bool first_iteration_flag = true;

	for (;;) {
		printf("update_gate_loop_begin \n");

		// Read Input Data
		for(int i=0; i< x_input_size; i++)
			x[i] = readincr(x_input, tlast);

		if (first_iteration_flag) {
			for(int i=0; i < h_input_size; i++)
				h[i] = 0;
			first_iteration_flag = false;
		}
		else {
			for(int i=0; i < h_input_size; i++)
				h[i] = readincr(h_input);
		}

		if (tlast) break;

		v8float* v8_x = (v8float*) &(x[0]);
		//Perform  multiplications
		// x Vector (1xd)x(dxh) Wxu Matrix
		alignas(32) v8float Wxu_x_res [h_vector_size];
		matrix_vec_mult <x_vector_size,h_vector_size> (v8_Wxu, v8_x , Wxu_x_res);

		v8float* v8_h = (v8float*) &(h[0]);
		// h Vector (1xh)x(hxh) Whu Matrix
		alignas(32) v8float Whu_x_res [h_vector_size];
		matrix_vec_mult <h_vector_size,h_vector_size> (v8_Whu, v8_h, Whu_x_res);

		v8float* v8_Wxu_x_res = (v8float*) &Wxu_x_res[0];
		v8float* v8_Whu_x_res = (v8float*) &Whu_x_res[0];

		//Add all element wise
		alignas(32) v8float v8_add[h_vector_size];
		for(int i=0; i<h_vector_size; i++){
			v8_add[i] = null_v8float();
			v8_add[i] = fpadd(v8_Wxu_x_res[i],
							v8_Whu_x_res[i]
							 );
			v8_add[i] = fpadd(v8_add[i],
							  v8_bu[i]
							  );
		}

		alignas(32) v8float sigm_C[h_input_size];
		act_func <h_vector_size> (v8_add, sigm_C, 2., 0.5, 12./32., -1./32.);

		//Element wise multiplication
		alignas(32) v8float elem_mul[h_vector_size];
		for(int i=0; i<h_vector_size; i++){
			elem_mul[i] = fpmul(sigm_C[i],
								0x0,
								0x76543210,
								v8_h[i],
								0x0,
								0x76543210
					);
		}

		// Converting to float pointers in order to write
		// can this be done without converting?
		float* U = (float*) &sigm_C;
		float* U_elem_mul_h = (float*) &elem_mul;

		for(int i=0; i<h_input_size; i++){
			writeincr(output, U[i]);
		}

		for(int i=0; i<h_input_size; i++){
			writeincr(output, U_elem_mul_h[i]);
		}
		printf("update_gate_loop_end \n");
	}
printf("update_gate_STOPED \n");
}
