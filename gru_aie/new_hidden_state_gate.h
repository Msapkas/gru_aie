#ifndef new_hidden_state_gate_h
#define new_hidden_state_gate_h

#include <adf.h>

//#define vector_size 8
//#define x_input_size 1
//#define h_input_size 8*vector_size

template<int x_input_size, int h_input_size>
void new_hidden_state_gate(
		input_stream<accfloat>* candidate_hidden_state_input,
		input_stream<float>* hidden_and_update_state_input,
		output_stream<accfloat>* new_hidden_state_output,
		output_stream<float>* hidden_feedback
		);

#endif
