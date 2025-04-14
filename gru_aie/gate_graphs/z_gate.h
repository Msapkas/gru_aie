#ifndef Z_GATE_H
#define Z_GATE_H

#include <adf.h>
#include "../mat_vec_mul/mat_input_vec_mul.h"
#include "../mat_vec_mul/mat_hidden_vec_mul.h"
#include "../adder/adder.h"
#include "../activation_functions/sigmoid.h"
#include "../config.h"
// #include "adf/new_frontend/types.h"

class z_gate: public adf::graph {
public:
    // Input
    adf::port<adf::input> x_input;
    adf::port<adf::input> hidden_input;
    // RTP to initialize the hidden state
    adf::port<adf::input> hidden_init;
    // Output
    adf::port<adf::output> z_output[VECTOR_LANES];

    // Declare kernels
    // ------------------------------
    adf::kernel Wz_x;
    adf::port<adf::input> Wz;
    
    adf::kernel Uz_h;
    adf::port<adf::input> Uz;

    adf::kernel add;
    adf::port<adf::input> identifier;
    adf::port<adf::input> bz;

    adf::pktsplit<VECTOR_LANES> vector_split;

    adf::kernel apply_sigmoid[VECTOR_LANES];
    // ------------------------------

    z_gate() {

    // Update (Z) Gate ------------------------------------------------------

    // Wz[N](rows) x x_input_vector
    Wz_x = adf::kernel::create(mat_input_vec_mul);
    adf::source(Wz_x) = "mat_vec_mul/mat_input_vec_mul.cc";
    adf::runtime<ratio>(Wz_x) = 1;

    adf::connect<adf::stream>(x_input, Wz_x.in[0]);
    adf::connect<adf::parameter>(Wz, adf::async(Wz_x.in[1]));

    // Uz[N](rows) x prev_hidden_vector
    Uz_h = adf::kernel::create(mat_hidden_vec_mul);
    adf::source(Uz_h) = "mat_vec_mul/mat_hidden_vec_mul.cc";
    adf::runtime<ratio>(Uz_h) = 1;

    adf::connect<adf::stream>(hidden_input, Uz_h.in[0]);
    adf::connect<adf::parameter>(Uz, adf::async(Uz_h.in[1]));
    adf::connect<adf::parameter>(hidden_init, adf::async(Uz_h.in[2]));

    // Add the outputs from the Matrix Vector input and hidden
    add = adf::kernel::create(adder);
    adf::source(add) = "adder/adder.cc";
    adf::runtime<ratio>(add) = 1;

    adf::connect<adf::stream>(Wz_x.out[0], add.in[0]);
    adf::connect<adf::stream>(Uz_h.out[0], add.in[1]);

    adf::connect<adf::parameter>(bz, adf::async(add.in[2]));
    adf::connect<adf::parameter>(identifier, adf::async(add.in[3]));

    // Split the vector to apply sigmoid element wise
    vector_split = adf::pktsplit<VECTOR_LANES>::create();
    adf::connect<adf::pktstream> (add.out[0], vector_split.in[0]);

    // Serve each output of the Adder - through the split - to a dedicated sigmoid kernel
    for (int i = 0; i < VECTOR_LANES; i++)
        {
        apply_sigmoid[i] = adf::kernel::create(sigmoid);
        adf::source(apply_sigmoid[i]) = "activation_functions/sigmoid.cc";
        adf::runtime<ratio>(apply_sigmoid[i]) = 1;
        
        adf::connect<adf::pktstream> (vector_split.out[i], apply_sigmoid[i].in[0]);

        // R Gate Elements Output 
        // The kernel outputs directly to the ports of the subgraph (4 in total)
        adf::connect<adf::pktstream> (apply_sigmoid[i].out[0], z_output[i]);
    }

    // ----------------------------------------------------------------------

    }

};

#endif
