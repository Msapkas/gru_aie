#ifndef BASIC_GATE_H
#define BASIC_GATE_H

#include <adf.h>
#include "./mat_vec_mul/mat_input_vec_mul.h"
#include "config.h"
#include <sstream> // For stringstream
#include <string>  // For string

class gru: public adf::graph {
public:
    adf::port<adf::input> x_input;
    adf::kernel W_x[H_VECTOR_SIZE/DIST_COEFF];
    adf::port<adf::input> W_weights[H_VECTOR_SIZE/DIST_COEFF];

    adf::input_plio plio_in;
    adf::output_plio plio_out[H_VECTOR_SIZE/DIST_COEFF];

    gru() {
        plio_in = adf::input_plio::create(adf::plio_128_bits, "data/test_in.txt");

        for (int i = 0; i < H_VECTOR_SIZE/DIST_COEFF; i++) {
            std::ostringstream filename;
            filename << "data/test_out_" << i << ".txt";  // Construct the filename with index i
            plio_out[i] = adf::output_plio::create(adf::plio_128_bits, filename.str());
        }

        for (int i = 0; i < H_VECTOR_SIZE/DIST_COEFF; i++) {
            W_x[i] = adf::kernel::create(mat_input_vec_mul);
            adf::connect(plio_in.out[0], W_x[i].in[0]);
            adf::connect(W_x[i].out[0], plio_out[i].in[0]);
            adf::connect<adf::parameter>(W_weights[i], adf::async(W_x[i].in[1]));
            adf::source(W_x[i]) = "mat_vec_mul/mat_input_vec_mul.cc";
            adf::runtime<ratio>(W_x[i]) = 1;
        }
    }
};

#endif
