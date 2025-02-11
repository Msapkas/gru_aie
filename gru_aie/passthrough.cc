#include "passthrough.h"

using namespace adf;

template<int h_input_size>
void passthrough(
		input_stream<float>* from_output,
		output_stream<float>* to_input
		) {

	bool tlast = false;
	float new_hidden[h_input_size];

	for (;;) {
		printf("pass_gate_loop_begin \n");

		for(int i=0; i<h_input_size; i++)
			new_hidden[i] = readincr(from_output, tlast);

		if (tlast) break;

		for(int i=0; i<h_input_size; i++)
			writeincr(to_input, new_hidden[i]);
		printf("pass_gate_loop_end \n");
	}
	printf("PASSTHROUGH_gate_terminated \n");
}
