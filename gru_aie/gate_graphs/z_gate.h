#ifndef Z_GATE_H
#define Z_GATE_H

#include <adf.h>
#include "../mat_vec_mul/mat_input_vec_mul.h"
#include "../mat_vec_mul/mat_hidden_vec_mul.h"
#include "../act_reduce/sigmoid_reduce.h"
#include "../config.h"

class z_gate: public adf::graph {
public:
    // Input
    adf::port<adf::input> x_input;
    adf::port<adf::input> hidden_input;
    // RTP to initialize the hidden state
    adf::port<adf::input> hidden_init;
    // Output
    adf::port<adf::output> z_output;

    // Declare kernels
    // ------------------------------
    adf::kernel Wz_x;
    adf::port<adf::input> Wz;
    
    adf::kernel Uz_h;
    adf::port<adf::input> Uz;

    adf::kernel z_sigm_reduce;
    adf::port<adf::input> identifier;
    adf::port<adf::input> bz;
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

    // Reduce_add and apply sigmoid LUT
    z_sigm_reduce = adf::kernel::create(sigmoid_reduce);
    adf::source(z_sigm_reduce) = "act_reduce/sigmoid_reduce.cc";
    adf::runtime<ratio>(z_sigm_reduce) = 1;

    adf::connect<adf::stream>(Wz_x.out[0], z_sigm_reduce.in[0]);
    adf::connect<adf::stream>(Uz_h.out[0], z_sigm_reduce.in[1]);
    adf::connect<adf::parameter>(bz, adf::async(z_sigm_reduce.in[2]));
    adf::connect<adf::parameter>(identifier, adf::async(z_sigm_reduce.in[3]));

    // Z Gate Elements Output
    adf::connect<adf::pktstream>(z_sigm_reduce.out[0], z_output);
    // -----------------------------------------------------------------------

    }

};

#endif
