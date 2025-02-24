#ifndef GRU_H
#define GRU_H

#include <adf.h>
#include "../gate_graphs/r_gate.h"
#include "../gate_graphs/z_gate.h"
// #include "../gate_graphs/candidate_hidden_state.h"
#include "../new_hidden_state_kernel/new_hidden_state.h"
#include "../config.h"

class gru : public adf::graph {
    public:

    adf::input_plio  PL_IN;
    adf::input_plio  PL_H;
    adf::output_plio PL_OUT_R;
    // adf::output_plio PL_OUT_Z;

    // Reset gate declarations
    r_gate r_gates[NKERNELS];
    adf::port<adf::input> r_hidden_initialization[NKERNELS];
    adf::port<adf::input> Wr_params[NKERNELS];
    adf::port<adf::input> Ur_params[NKERNELS];
    adf::port<adf::input> br_params[NKERNELS];
    
    adf::pktmerge<NKERNELS> r_aggregator;

    // // Update gate declarations
    // z_gate z_gates[H_VECTOR_SIZE/DIST_COEFF];
    // adf::port<adf::input> z_hidden_initialization[H_VECTOR_SIZE/DIST_COEFF];
    // adf::port<adf::input> Wz_params[H_VECTOR_SIZE/DIST_COEFF];
    // adf::port<adf::input> Uz_params[H_VECTOR_SIZE/DIST_COEFF];
    // adf::port<adf::input> bz_params[H_VECTOR_SIZE/DIST_COEFF];

    // adf::pktmerge<H_VECTOR_SIZE/DIST_COEFF> z_aggregator;

    // // Cand Hidden state Gate decs
    // candidate_hidden_gate candidate_hidden_gates[H_VECTOR_SIZE/DIST_COEFF];
    // adf::port<adf::input> Wh_params[H_VECTOR_SIZE/DIST_COEFF];
    // adf::port<adf::input> Uh_params[H_VECTOR_SIZE/DIST_COEFF];
    // adf::port<adf::input> bh_params[H_VECTOR_SIZE/DIST_COEFF];

    // // New hidden state Gate
    // adf::kernel new_h_gate;

    gru () {

        PL_IN = adf::input_plio::create(adf::plio_128_bits, "data/test_in.txt");
        PL_H = adf::input_plio::create(adf::plio_128_bits, "data/test_in.txt");
        PL_OUT_R = adf::output_plio::create(adf::plio_128_bits,"data/r_outputs.txt");
        // PL_OUT_Z = adf::output_plio::create(adf::plio_128_bits,"data/z_outputs.txt");

        r_aggregator = adf::pktmerge<NKERNELS>::create();
        // z_aggregator = adf::pktmerge<H_VECTOR_SIZE/DIST_COEFF>::create();

        for (int i = 0; i < NKERNELS; i++){
            
            // R gates connections
            adf::connect<> (PL_IN.out[0], r_gates[i].x_input);

            adf::connect<adf::parameter> (PL_H.out[0], r_gates[i].hidden_input);
            adf::connect<adf::parameter> (r_hidden_initialization[i], adf::async(r_gates[i].hidden_init));

            adf::connect<adf::parameter> (Wr_params[i], adf::async(r_gates[i].Wr));
            adf::connect<adf::parameter> (Ur_params[i], adf::async(r_gates[i].Ur));
            adf::connect<adf::parameter> (br_params[i], r_gates[i].br);

            adf::connect<adf::pktstream> (r_gates[i].r_output, r_aggregator.in[i]);

            // Z gate connections
            // adf::connect<> (PL_IN.out[0], z_gates[i].x_input);

            // adf::connect<adf::parameter> (PL_H.out[0], z_gates[i].hidden_input);
            // adf::connect<adf::parameter> (z_hidden_initialization[i], adf::async(z_gates[i].hidden_init));

            // adf::connect<adf::parameter> (Wz_params[i], adf::async(z_gates[i].Wz));
            // adf::connect<adf::parameter> (Uz_params[i], adf::async(z_gates[i].Uz));
            // adf::connect<adf::parameter> (bz_params[i], z_gates[i].bz);
            // adf::connect<adf::pktstream> (z_gates[i].z_output, z_aggregator.in[i]);

            // New Hidden state connections
            

        }

        adf::connect<> (r_aggregator.out[0], PL_OUT_R.in[0]);
        // adf::connect<> (z_aggregator.out[0], PL_OUT_Z.in[0]);

    }
};

#endif