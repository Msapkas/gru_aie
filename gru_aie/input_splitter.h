#ifndef input_splitter_h
#define input_splitter_h

#include <adf.h>

template<int x_input_size>
void input_splitter(
		input_stream<accfloat>* circ_buff_input,
		output_stream<accfloat>* to_rst_cnd_gates,
		output_stream<float>* to_upd_gate
		);

#endif
