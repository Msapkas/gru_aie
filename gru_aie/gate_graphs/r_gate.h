#ifndef R_GATE_H
#define R_GATE_H

#include <adf.h>
#include "../mat_vec_mul/mat_input_vec_mul.h"
#include "../mat_vec_mul/mat_hidden_vec_mul.h"
#include "../act_reduce/sigmoid_reduce.h"
#include "../config.h"

class r_gate: public adf::graph {
public:
    // Input
    adf::port<adf::input> x_input;
    adf::port<adf::input> hidden_input;
    // RTP to initialize the hidden state
    adf::port<adf::input> hidden_init;
    // Output
    adf::port<adf::output> r_output;

    // Declare kernels
    // ------------------------------
    adf::kernel Wr_x;
    adf::port<adf::input> Wr;

    adf::kernel Ur_h;
    adf::port<adf::input> Ur;

    adf::kernel r_sigm_reduce;
    adf::port<adf::input> identifier;
    adf::port<adf::input> br;
    // ------------------------------

    r_gate() {

    // Reset Gate (R) -------------------------------------------------------

    // Wr[N](rows) x x_input_vector
    Wr_x = adf::kernel::create(mat_input_vec_mul);
    adf::source(Wr_x) = "mat_vec_mul/mat_input_vec_mul.cc";
    adf::runtime<ratio>(Wr_x) = 1;

    adf::connect<adf::stream>(x_input, Wr_x.in[0]);
    adf::connect<adf::parameter>(Wr, adf::async(Wr_x.in[1]));

    // Ur[N](rows) x prev_hidden_vector
    Ur_h = adf::kernel::create(mat_hidden_vec_mul);
    adf::source(Ur_h) = "mat_vec_mul/mat_hidden_vec_mul.cc";
    adf::runtime<ratio>(Ur_h) = 1;

    adf::connect<adf::stream>(hidden_input, Ur_h.in[0]);
    adf::connect<adf::parameter>(Ur, adf::async(Ur_h.in[1]));
    adf::connect<adf::parameter>(hidden_init, adf::async(Ur_h.in[2]));

    // Reduce_add and apply sigmoid LUT
    r_sigm_reduce = adf::kernel::create(sigmoid_reduce);
    adf::source(r_sigm_reduce) = "act_reduce/sigmoid_reduce.cc";
    adf::runtime<ratio>(r_sigm_reduce) = 1;

    adf::connect<adf::stream>(Wr_x.out[0], r_sigm_reduce.in[0]);
    adf::connect<adf::stream>(Ur_h.out[0], r_sigm_reduce.in[1]);
    adf::connect<adf::parameter>(br, adf::async(r_sigm_reduce.in[2]));
    adf::connect<adf::parameter>(identifier, adf::async(r_sigm_reduce.in[3]));

    // R Gate Elements Output
    adf::connect<adf::pktstream>(r_sigm_reduce.out[0], r_output);
    // ----------------------------------------------------------------------

    }

};

#endif
