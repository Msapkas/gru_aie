#include "new_hidden_state_gate.h"

using namespace adf;

template<int x_input_size, int h_input_size>
void new_hidden_state_gate(
		input_stream<accfloat>* candidate_hidden_state_input,
		input_stream<float>* hidden_and_update_state_input,
		output_stream<accfloat>* new_hidden_state_output,
		output_stream<float>* hidden_feedback
		) {

	const int h_vector_size = h_input_size/8;

	static v8float Cand_hdn_s[h_vector_size];
	static float one_minus_Upd_s[h_input_size];
	static float U_elem_mul_h_output[h_input_size];

	v8float cmd_buffer = null_v8float();
	// Message to stop next stage
	v8float stop_msg = null_v8float();
	stop_msg = upd_elem(stop_msg, 0, 1.);

	for (;;){

		printf("new_gate_loop_begin \n");

		for(int i=0; i < h_vector_size; i++)
			Cand_hdn_s[i] = readincr_v8(candidate_hidden_state_input);
		cmd_buffer = readincr_v8(candidate_hidden_state_input);
		if (ext_elem(cmd_buffer, 0) != 0) break;

		// Read input Data make sure is in the same order with the write of the update gate
		for(int i=0; i < h_input_size; i++)
			one_minus_Upd_s[i] = 1. - readincr(hidden_and_update_state_input);

		for(int i=0; i < h_input_size; i++)
			U_elem_mul_h_output[i] = readincr(hidden_and_update_state_input);

		//

		v8float* v8_one_minus_Upd_s = (v8float*) &one_minus_Upd_s[0];
//		v8float* restrict v8_Cand_hdn_s = (v8float*) &Cand_hdn_s[0];

		//Element wise multiplication
		alignas(32) v8float v8_elem_mul[h_vector_size];
		for(int i=0; i<h_vector_size; i++){
			v8_elem_mul[i] = fpmul(v8_one_minus_Upd_s[i],
								0x0,
								0x76543210,
								Cand_hdn_s[i],
								0x0,
								0x76543210
							 );
		}

		//Add all element wise
		v8float* v8_U_elem_mul_h_output = (v8float*) &U_elem_mul_h_output[0];
		alignas(32) v8float v8_new_hidden[h_vector_size];
		for(int i=0; i<h_vector_size; i++){
			v8_new_hidden[i] = fpadd(v8_elem_mul[i],
									 v8_U_elem_mul_h_output[i]
									);
		}

//		float *new_hidden = (float*) &v8_new_hidden;

		for (int i=0; i<h_vector_size; i++)
			writeincr_v8(new_hidden_state_output, v8_new_hidden[i]);

		writeincr_v8(new_hidden_state_output, cmd_buffer);

		float* new_hidden = (float*) &v8_new_hidden[0]; //HERE CHECK

		for (int i=0; i<h_input_size; i++)
			writeincr(hidden_feedback, new_hidden[i]);

		printf("new_gate_loop_end \n");
	}

	printf("exited_NEWHIDSTS_loop_writing_dummy_outputs \n");

	for(int i=0; i<h_vector_size; i++)
		writeincr_v8(new_hidden_state_output, null_v8float());

	printf("exited_NEWHIDSTS_loop_writing_STOPCOMMAND \n");

	writeincr_v8(new_hidden_state_output, stop_msg);

	printf("exited_NEWHIDSTS_loop_STOPCOMMAND_wrote_writing_stream \n");

	for (int i=0; i<h_input_size; i++)
		writeincr(hidden_feedback, 0., true);

	printf("exited_NEWHIDSTS_loop_stream_hidden_wrote \n");

}
