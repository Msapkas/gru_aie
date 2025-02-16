#ifndef GRU_H
#define GRU_H

#include <adf.h>
#include "../gate_graphs/r_gate.h"
#include "../config.h"

class gru : public adf::graph {
    public:

    adf::input_plio  PL_IN;
    adf::input_plio  PL_H;
    adf::output_plio PL_OUT;

    r_gate r_gates[H_VECTOR_SIZE/DIST_COEFF];

    adf::port<adf::input> hidden_initialization[H_VECTOR_SIZE/DIST_COEFF];
    adf::port<adf::input> W_params[H_VECTOR_SIZE/DIST_COEFF];
    adf::port<adf::input> U_params[H_VECTOR_SIZE/DIST_COEFF];
    adf::port<adf::input> b_params[H_VECTOR_SIZE/DIST_COEFF];

    adf::pktmerge<H_VECTOR_SIZE/DIST_COEFF> r_aggregator;

    gru () {

        PL_IN = adf::input_plio::create(adf::plio_128_bits, "data/test_in.txt");
        PL_OUT = adf::output_plio::create(adf::plio_128_bits,"data/outputs.txt");
        PL_H = adf::input_plio::create(adf::plio_128_bits, "data/test_in.txt");

        r_aggregator = adf::pktmerge<H_VECTOR_SIZE/DIST_COEFF>::create();

        for (int i = 0; i < H_VECTOR_SIZE/DIST_COEFF; i++){
            adf::connect<> (PL_IN.out[0], r_gates[i].x_input);
            adf::connect<adf::parameter> (PL_H.out[0], r_gates[i].hidden_input);
            adf::connect<adf::parameter> (hidden_initialization[i], adf::async(r_gates[i].hidden_init));

            adf::connect<adf::parameter> (W_params[i], adf::async(r_gates[i].Wr));
            adf::connect<adf::parameter> (U_params[i], adf::async(r_gates[i].Ur));
            adf::connect<adf::parameter> (b_params[i], adf::async(r_gates[i].br));
            adf::connect<adf::pktstream> (r_gates[i].r_output, r_aggregator.in[i]);
        }

        adf::connect<> (r_aggregator.out[0], PL_OUT.in[0]);
        
    }
};

#endif