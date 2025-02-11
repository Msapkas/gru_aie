#include "reset_and_candidate_gates.h"
#include "matrix_vec_mult.h"
#include "act_func.h"

using namespace adf;

#ifdef __X86SIM__
#include <fstream>
#include <string>
void print_to_file(std::ofstream &f, std::string label, v8float *buf, int len, int counter) {
	f << label << " " << counter << ":\n";
	for (int i = 0; i< len; i++) {
		for (int j = 0; j < 8; j++) {
			f << ext_elem(buf[i], j) << " ";
		}
	}
	f << "\n";
}
#endif

template<int x_input_size, int h_input_size, int x_vector_size, int h_vector_size>
void reset_and_candidate_gates(
		input_stream<accfloat>* x_input,
		input_stream<float>* h_input,
		output_stream<accfloat>* candidate_hidden_state,

		const float (&weights_Whr)[h_input_size*h_input_size],
		const float (&weights_Wxr)[x_input_size*h_input_size],
		const float (&biases_r)[h_input_size],

		const float (&weights_Whh)[h_input_size*h_input_size],
		const float (&weights_Wxh)[x_input_size*h_input_size],
		const float (&biases_h)[h_input_size]

		){

//	const int x_vector_size = x_input_size/8;
//	const int h_vector_size = h_input_size/8;

	#ifdef __X86SIM__
	std::ofstream f("activation_tracing.txt");
	int counter = 0;
	#endif

	static v8float x[x_vector_size];
	alignas(32) static float h[h_input_size];

	v8float* v8_Wxr = (v8float*) &(weights_Wxr[0]);
	v8float* v8_Whr = (v8float*) &(weights_Whr[0]);
	v8float* v8_br = (v8float*) &(biases_r[0]);

	v8float* v8_Wxh = (v8float*) &(weights_Wxh[0]);
	v8float* v8_Whh = (v8float*) &(weights_Whh[0]);
	v8float* v8_bh = (v8float*) &(biases_h[0]);

	v8float cmd_buffer = null_v8float();
	// Message to stop next stage
	v8float stop_msg = null_v8float();
	stop_msg = upd_elem(stop_msg, 0, 1.);

	bool first_iteration_flag = true;

	for (;;) {
//		printf("reset_gate_loop_begin \n");
		// Read Input Data
		for(int i=0; i< x_vector_size; i++)
			x[i] = readincr_v8(x_input);

		cmd_buffer = readincr_v8(x_input);
		if (ext_elem(cmd_buffer, 0) != 0) break;

		if (first_iteration_flag) {
			for(int i=0; i < h_input_size; i++)
				h[i] = 0.;
			first_iteration_flag = false;
		}
		else {
			for(int i=0; i < h_input_size; i++)
				h[i] = readincr(h_input);
		}
		// Begin of Reset Gate //

		//Perform  multiplications
		// x Vector (1xd)x(dxh) Wxr Matrix
		alignas(32) v8float Wxr_x_res [h_vector_size];
		matrix_vec_mult <x_vector_size,h_vector_size> (v8_Wxr, x , Wxr_x_res);

		v8float* v8_h = (v8float*) &(h[0]);

		alignas(32) v8float Whr_x_res [h_vector_size];
		matrix_vec_mult<h_vector_size,h_vector_size>(v8_Whr, v8_h , Whr_x_res);

		v8float* v8_Wxr_x_res = (v8float*) &Wxr_x_res[0];
		v8float* v8_Whr_x_res = (v8float*) &Whr_x_res[0];

		//Add all element wise
		alignas(32) v8float v8_add[h_vector_size];
		for(int i=0; i<h_vector_size; i++){
			v8_add[i] = null_v8float();
			v8_add[i] = fpadd(v8_Wxr_x_res[i],
							v8_Whr_x_res[i]
							 );
			v8_add[i] = fpadd(v8_add[i],
							  v8_br[i]
							  );
		}

		//Apply sigmoid
		alignas(32) v8float sigm_R[h_vector_size];

		#ifdef __X86SIM__
		print_to_file(f, "input", v8_add, h_vector_size, counter);
		#endif

		act_func <h_vector_size> (v8_add, sigm_R, 2., 0.5, 12./32., -1./32.);

		#ifdef __X86SIM__
		print_to_file(f, "output", sigm_R, h_vector_size, counter);
		counter++;
		#endif

		// Begin of Candidate Hidden State Gate //

		//Perform  multiplications
		// x Vector (1xd)x(dxh) Wxh Matrix
		alignas(32) v8float Wxh_x_res [h_vector_size];
		matrix_vec_mult<x_vector_size,h_vector_size>(v8_Wxh, x , Wxh_x_res);

		//Element wise multiplication
		alignas(32) v8float elem_mul[h_vector_size];
		for(int i=0; i<h_vector_size; i++){
			elem_mul[i] = 	fpmul(sigm_R[i],
								0x0,
								0x76543210,
								v8_h[i],
								0x0,
								0x76543210
					);
		}

		// ElemWiseResult (1xh)x(hxh) Whh Matrix
		alignas(32) v8float Whh_Rxh_res [h_vector_size];
		matrix_vec_mult<h_vector_size,h_vector_size>(v8_Whh, elem_mul, Whh_Rxh_res);

		v8float* v8_Wxh_x_res = (v8float*) &Wxh_x_res[0];
		v8float* v8_Whh_Rxh_res = (v8float*) &Whh_Rxh_res[0];

		//Add all element wise
		//v8_add was defined previously
		for(int i=0; i<h_vector_size; i++){
			v8_add[i] = null_v8float();
			v8_add[i] = fpadd(v8_Wxh_x_res[i],
							v8_Whh_Rxh_res[i]
							 );
			v8_add[i] = fpadd(v8_add[i],
							  v8_bh[i]
							  );
		}

		//Apply Tanh
		alignas(32) v8float v8_tanh[h_vector_size];
		act_func<h_vector_size>(v8_add, v8_tanh, 2., 0., 12./16., -1./16.);

		for(int i=0; i<h_vector_size; i++)
			writeincr_v8(candidate_hidden_state, v8_tanh[i]);

		writeincr_v8(candidate_hidden_state, cmd_buffer);
//		printf("reset_gate_loop_end \n");
	}

//	printf("exited_RESETnCAND_loop_writing_dummy_outputs \n");
	for(int i=0; i<h_vector_size; i++)
		writeincr_v8(candidate_hidden_state, null_v8float());

//	printf("exited_RESETnCAND_loop_writing_STOPCOMMAND \n");
	writeincr_v8(candidate_hidden_state, stop_msg);
	printf("exited_RESETnCAND_loop_STOPCOMMAND_wrote \n");

	for(int i=0; i < h_input_size; i++)
		h[i] = readincr(h_input);

	printf("exited_RESETnCAND_loop_final_read_h \n");

}
