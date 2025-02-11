#ifndef update_gate_h
#define update_gate_h

#include <adf.h>
#include <aie_api/aie.hpp>

//#define vector_size 8
//#define x_input_size 1
//#define h_input_size 8*vector_size

template<int x_input_size, int h_input_size>
void update_gate(
		input_stream<float>* x_input,
		input_stream<float>* h_input,

		output_stream<float>* output,

		const float (&weights_Whu)[h_input_size*h_input_size],
		const float (&weights_Wxu)[x_input_size*h_input_size],
		const float (&biases_u)[h_input_size]

		);

#endif
