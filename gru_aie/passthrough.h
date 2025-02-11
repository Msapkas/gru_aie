#ifndef passthrough_h
#define passthrough_h

#include <adf.h>

//#define vector_size 8
//#define h_input_size 8*vector_size

template<int h_input_size>
void passthrough(
		input_stream<float>* from_output,
		output_stream<float>* to_input
		);

#endif
