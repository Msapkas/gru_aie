#include "input_splitter.h"

#define vector_size 3

template<int x_input_size>
void input_splitter(
		input_stream<accfloat>* circ_buff_input,
		output_stream<accfloat>* to_rst_cnd_gates,
		output_stream<float>* to_upd_gate
		){

	v8float x_input[vector_size];
	v8float cmd_buffer = null_v8float();

	// Message to stop next stage
	v8float stop_msg = null_v8float();
	stop_msg = upd_elem(stop_msg, 0, 1.);

	for (;;){
//		printf("input_gate_loop_begin \n");

		// Read X from circular buffer
		for(int i=0; i<vector_size; i++)
			x_input[i] = readincr_v8(circ_buff_input);

//		printf("input_gate_read_x \n");

		cmd_buffer = readincr_v8(circ_buff_input);
		if (ext_elem(cmd_buffer, 0) != 0) break;

		// Read complete
		// Write cascade to the reset gate + cmd
//		printf("input_gate_read_cmd_buff \n");

		for(int i=0; i<vector_size; i++)
			writeincr_v8(to_rst_cnd_gates, x_input[i]);

		writeincr_v8(to_rst_cnd_gates, cmd_buffer);
		//
//		printf("input_gate_wrote_x_to_reset \n");

		float* float_x_input = (float*) &x_input;

		for(int i=0; i<x_input_size; i++)
			writeincr(to_upd_gate, float_x_input[i]);

//		printf("input_gate_wrote_x_to_update \n");

//		printf("input_gate_loop_end \n");
	}

//	printf("exited_SPLITTER_loop_writing_dummy_outputs \n");

	for(int i=0; i<vector_size; i++)
		writeincr_v8(to_rst_cnd_gates, null_v8float());

//	printf("exited_SPLITTER_loop_writing_STOPCOMMAND \n");

	writeincr_v8(to_rst_cnd_gates, stop_msg);

//	printf("exited_SPLITTER_loop_STOPCOMMAND_wrote_writing_stream \n");

	for(int i=0; i<x_input_size; i++)
		writeincr(to_upd_gate, 0., true);

	printf("exited_SPLITTER_loop_stream_hidden_wrote \n");

}
