#ifndef CANDIDATE_HIDDEN_STATE 
#define CANDIDATE_HIDDEN_STATE

#include <adf.h>
#include "../mat_vec_mul/mat_input_vec_mul.h"
#include "../mat_vec_mul/chsg_mat_r_mul_h.h"
#include "../adder/adder.h"
#include "../activation_functions/hyper_tan.h"
#include "../config.h"

class candidate_hidden_gate: public adf::graph {
public:
    // Input
    adf::port<adf::input> x_input;
    adf::port<adf::input> r_input;
    adf::port<adf::input> hidden_input;
    // RTP to initialize the hidden state
    adf::port<adf::input> hidden_init;
    // Output
    adf::port<adf::output> chsg_output[VECTOR_LANES];

    // Declare kernels
    // ------------------------------
    adf::kernel Wh_x;
    adf::port<adf::input> Wh;

    adf::kernel Uh_h;
    adf::port<adf::input> Uh;

    adf::kernel add; //
    adf::port<adf::input> identifier;
    adf::port<adf::input> bh;

    adf::pktsplit<VECTOR_LANES> vector_split;

    adf::kernel apply_tanh[VECTOR_LANES];
    // ------------------------------

    candidate_hidden_gate () {

    // Candidate Hidden State Gate (Hhat) -------------------------------------------------------

    // Wh[N](rows) x x_input_vector
    Wh_x = adf::kernel::create(mat_input_vec_mul);
    adf::source(Wh_x) = "mat_vec_mul/mat_input_vec_mul.cc";
    adf::runtime<ratio>(Wh_x) = 1;

    adf::connect<>(x_input, Wh_x.in[0]);
    adf::connect<adf::parameter>(Wh, adf::async(Wh_x.in[1]));

    // Uh[N](rows) x prev_hidden_vector
    Uh_h = adf::kernel::create(chsg_mat_r_mul_h);
    adf::source(Uh_h) = "mat_vec_mul/chsg_mat_r_mul_h.cc";
    adf::runtime<ratio>(Uh_h) = 1;

    adf::connect<>(r_input, Uh_h.in[0]);
    adf::connect<>(hidden_input, Uh_h.in[1]);
    adf::connect<adf::parameter>(Uh, adf::async(Uh_h.in[2]));
    adf::connect<adf::parameter>(hidden_init, adf::async(Uh_h.in[3]));

    // Add the outputs from the Matrix Vector input and hidden
    add = adf::kernel::create(adder);
    adf::source(add) = "adder/adder.cc";
    adf::runtime<ratio>(add) = 1;

    adf::connect<adf::stream>(Wh_x.out[0], add.in[0]);
    adf::connect<adf::stream>(Uh_h.out[0], add.in[1]);

    adf::connect<adf::parameter>(bh, adf::async(add.in[2]));
    adf::connect<adf::parameter>(identifier, adf::async(add.in[3]));

    // Split the vector to apply sigmoid element wise
    vector_split = adf::pktsplit<VECTOR_LANES>::create();
    adf::connect<adf::pktstream> (add.out[0], vector_split.in[0]);

    // Serve each output of the Adder - through the split - to a dedicated sigmoid kernel
    for (int i = 0; i < VECTOR_LANES; i++)
        {
        apply_tanh[i] = adf::kernel::create(hyper_tan);
        adf::source(apply_tanh[i]) = "activation_functions/hyper_tan.cc";
        adf::runtime<ratio>(apply_tanh[i]) = 1;
        
        adf::connect<adf::pktstream> (vector_split.out[i], apply_tanh[i].in[0]);

        // R Gate Elements Output 
        // The kernel outputs directly to the ports of the subgraph (4 in total)
        adf::connect<adf::pktstream> (apply_tanh[i].out[0], chsg_output[i]);
    }

    // ----------------------------------------------------------------------
    }
};

#endif