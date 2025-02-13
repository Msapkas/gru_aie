#ifndef distr_mat_vec_mul_graph_h
#define distr_mat_vec_mul_graph_h

#include "config.h"
#include <adf.h>
#include "./Mat_vec_mul/mat_vec_mul.h"
#include <sstream> // For stringstream
#include <string>  // For string

class gru: public adf::graph {
public:
    adf::port<adf::input> x_input;
    adf::kernel w_v[H_SIZE/DIST_COEFF];
    adf::port<adf::input> dist_weights[H_SIZE/DIST_COEFF];

    adf::input_plio plio_in;
    adf::output_plio plio_out[H_SIZE/DIST_COEFF];

    gru() {
        plio_in = adf::input_plio::create(adf::plio_128_bits, "data/test_in.txt");

        for (int i = 0; i < H_SIZE / DIST_COEFF; i++) {
            std::ostringstream filename;
            filename << "data/test_out_" << i << ".txt";  // Construct the filename with index i
            plio_out[i] = adf::output_plio::create(adf::plio_128_bits, filename.str());
        }

        for (int i = 0; i < H_SIZE/DIST_COEFF; i++) {
            w_v[i] = adf::kernel::create(mat_vec_mul);
            adf::connect(plio_in.out[0], w_v[i].in[0]);
            adf::connect(w_v[i].out[0], plio_out[i].in[0]);
            adf::connect<adf::parameter>(dist_weights[i], adf::async(w_v[i].in[1]));
            adf::source(w_v[i]) = "Mat_vec_mul/mat_vec_mul.cc";
            adf::runtime<ratio>(w_v[i]) = 1;
        }
    }
};

#endif
