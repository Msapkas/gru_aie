#ifndef reset_and_candidate_gates_h
#define reset_and_candidate_gates_h

#include <adf.h>

//#define vector_size 8
//#define x_input_size 1
//#define h_input_size 8*vector_size

template<int x_input_size, int h_input_size, int x_vector_size=x_input_size/8, int h_vector_size=h_input_size/8>
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

		);

#endif
