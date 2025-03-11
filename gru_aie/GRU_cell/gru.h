#ifndef GRU_H
#define GRU_H

#include <adf.h>
#include "adf/new_frontend/types.h"

#include "../gate_graphs/r_gate.h"
#include "../aggregator_kernels/aggregator.h"
#include "../gate_graphs/z_gate.h"
#include "../gate_graphs/candidate_hidden_state.h"
#include "../new_hidden_state_kernel/new_hidden_state.h"
#include "../config.h"


class gru : public adf::graph {
    public:

    // // Pl IO
    adf::input_plio PL_INPUT;
    adf::output_plio PL_OUTPUT;

    // Reset gate declarations
    r_gate r_gates[NKERNELS];
    adf::port<adf::input> r_identifier[NKERNELS];
    adf::port<adf::input> r_hidden_initialization[NKERNELS];
    adf::port<adf::input> Wr_params[NKERNELS];
    adf::port<adf::input> Ur_params[NKERNELS];
    adf::port<adf::input> br_params[NKERNELS];
    adf::pktmerge<NKERNELS> r_merge;
    adf::kernel r_aggregator_kernel;

    // Update gate declarations
    z_gate z_gates[NKERNELS];
    adf::port<adf::input> z_identifier[NKERNELS];
    adf::port<adf::input> z_hidden_initialization[NKERNELS];
    adf::port<adf::input> Wz_params[NKERNELS];
    adf::port<adf::input> Uz_params[NKERNELS];
    adf::port<adf::input> bz_params[NKERNELS];
    adf::pktmerge<NKERNELS> z_merge;
    adf::kernel z_aggregator_kernel;

    // Cand Hidden state Gate decs
    candidate_hidden_gate candidate_hidden_gates[NKERNELS];
    adf::port<adf::input> chsg_identifier[NKERNELS];
    adf::port<adf::input> chsg_hidden_initialization[NKERNELS];
    adf::port<adf::input> Wh_params[NKERNELS];
    adf::port<adf::input> Uh_params[NKERNELS];
    adf::port<adf::input> bh_params[NKERNELS];
    adf::pktmerge<NKERNELS> chsg_merge;
    adf::kernel chsg_aggregator_kernel;

    // // // New hidden state Gate
    adf::kernel new_hidden_state_gate;
    adf::port<adf::input> new_hidden_state_gate_hidden_initialization;

    gru () {

        // PL I/O
        PL_INPUT = adf::input_plio::create(adf::plio_128_bits, "data/test_in_x.txt");
        PL_OUTPUT = adf::output_plio::create(adf::plio_128_bits,"data/outputs.txt");


        // pkg merges
        r_merge = adf::pktmerge<NKERNELS>::create();
        z_merge = adf::pktmerge<NKERNELS>::create();
        chsg_merge = adf::pktmerge<NKERNELS>::create();

        // Instatiate distributed Matrix - Vector Multiplications
        for (int i = 0; i < NKERNELS; i++){
            // ------------------------------
            // R gates connections
            adf::connect<> (PL_INPUT.out[0], r_gates[i].x_input);

            adf::connect<adf::parameter> (r_identifier[i], r_gates[i].identifier);

            adf::connect<adf::parameter> (r_hidden_initialization[i], r_gates[i].hidden_init);

            adf::connect<adf::parameter> (Wr_params[i], r_gates[i].Wr);
            adf::connect<adf::parameter> (Ur_params[i], r_gates[i].Ur);
            adf::connect<adf::parameter> (br_params[i], r_gates[i].br);

            adf::connect<adf::pktstream> (r_gates[i].r_output, r_merge.in[i]);

            // // ------------------------------
            // Z gate connections
            adf::connect<> (PL_INPUT.out[0], z_gates[i].x_input);

            adf::connect<adf::parameter> (z_identifier[i], z_gates[i].identifier);

            adf::connect<adf::parameter> (z_hidden_initialization[i], z_gates[i].hidden_init);

            adf::connect<adf::parameter> (Wz_params[i], z_gates[i].Wz);
            adf::connect<adf::parameter> (Uz_params[i], z_gates[i].Uz);
            adf::connect<adf::parameter> (bz_params[i], z_gates[i].bz);
            adf::connect<adf::pktstream> (z_gates[i].z_output, z_merge.in[i]);

            // // ------------------------------
            // Cand Hidden state connections
            adf::connect<> (PL_INPUT.out[0], candidate_hidden_gates[i].x_input);

            adf::connect<adf::parameter> (chsg_identifier[i], candidate_hidden_gates[i].identifier);

            adf::connect<adf::parameter> (chsg_hidden_initialization[i], candidate_hidden_gates[i].hidden_init);

            adf::connect<adf::parameter> (Wh_params[i], candidate_hidden_gates[i].Wh);
            adf::connect<adf::parameter> (Uh_params[i], candidate_hidden_gates[i].Uh);
            adf::connect<adf::parameter> (bh_params[i], candidate_hidden_gates[i].bh);
            adf::connect<adf::pktstream> (candidate_hidden_gates[i].cand_h_output, chsg_merge.in[i]);

        }
        
        // ------------------------------
        // R aggregator
        r_aggregator_kernel = adf::kernel::create(aggregator);
        adf::source(r_aggregator_kernel) = "aggregator_kernels/aggregator.cc";
        adf::runtime<ratio>(r_aggregator_kernel) = 1;

        adf::connect<adf::pktstream> (r_merge.out[0], r_aggregator_kernel.in[0]);

        // Connect the output to all the Candidate Hidden state Gates
        for (int i = 0; i < NKERNELS; i++){
            adf::connect<adf::stream> (r_aggregator_kernel.out[0], candidate_hidden_gates[i].r_input);
        }

        // // ------------------------------
        // Z aggregator could be redundant
        z_aggregator_kernel = adf::kernel::create(aggregator);
        adf::source(z_aggregator_kernel) = "aggregator_kernels/aggregator.cc";
        adf::runtime<ratio>(z_aggregator_kernel) = 1;

        adf::connect<adf::pktstream> (z_merge.out[0], z_aggregator_kernel.in[0]);

        // // ------------------------------
        // Cand Hidden State outputs aggregator
        chsg_aggregator_kernel = adf::kernel::create(aggregator);
        adf::source(chsg_aggregator_kernel) = "aggregator_kernels/aggregator.cc";
        adf::runtime<ratio>(chsg_aggregator_kernel) = 1;

        adf::connect<adf::pktstream> (chsg_merge.out[0], chsg_aggregator_kernel.in[0]);

        // // ------------------------------
        // New hidden state gate
        new_hidden_state_gate = adf::kernel::create(new_hidden_state);
        adf::source(new_hidden_state_gate) = "new_hidden_state_kernel/new_hidden_state.cc";
        adf::runtime<ratio>(new_hidden_state_gate) = 1;

        // Aggregated inputs
        adf::connect<adf::stream> (chsg_aggregator_kernel.out[0], new_hidden_state_gate.in[0]);
        adf::connect<adf::stream> (z_aggregator_kernel.out[0], new_hidden_state_gate.in[1]);

        adf::connect<adf::parameter>(new_hidden_state_gate_hidden_initialization, new_hidden_state_gate.in[2]);


        // New Hidden state Feedback
        for (int i = 0; i < NKERNELS; i++){
            adf::connect<adf::stream> (new_hidden_state_gate.out[0], r_gates[i].hidden_input);
            adf::connect<adf::stream> (new_hidden_state_gate.out[0], z_gates[i].hidden_input);
            adf::connect<adf::stream> (new_hidden_state_gate.out[0], candidate_hidden_gates[i].hidden_input);
        }

        // ------------------------------
        // New Hidden State PL Output
        // The OUPUT is sharing a blockign condition with ALOT of kernel. It may be wise to 
        // create an indipendent passthrough output kernel
        adf::connect<adf::stream> (new_hidden_state_gate.out[0], PL_OUTPUT.in[0]);

    }
};

#endif