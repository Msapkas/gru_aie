#ifndef CANDIDATE_HIDDEN_STATE 
#define CANDIDATE_HIDDEN_STATE

class candidate_hidden_gate: public adf::graph {
public:    
    // Input
    adf::port<adf::input> x_input;
    adf::port<adf::input> hidden_input;
    // RTP to initialize the hidden state
    adf::port<adf::input> hidden_init;
    // Output
    adf::port<adf::output> cand_h_output;

    // Declare kernels
    // ------------------------------
    adf::kernel Wh_x;
    adf::port<adf::input> Wh;
    adf::kernel Uh_h;
    adf::port<adf::input> Uh;
    adf::kernel tanh_reduce;
    adf::port<adf::input> bh;
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
    Uh_h = adf::kernel::create(mat_hidden_vec_mul);
    adf::source(Uh_h) = "mat_vec_mul/mat_hidden_vec_mul.cc";
    adf::runtime<ratio>(Uh_h) = 1;

    adf::connect<>(hidden_input, Uh_h.in[0]);
    adf::connect<adf::parameter>(Uh, adf::async(Uh_h.in[1]));
    adf::connect<adf::parameter>(hidden_init, adf::async(Uh_h.in[2]));

    // Reduce_add and apply sigmoid LUT
    tanh_reduce = adf::kernel::create(tanh_reduce);
    adf::source(tanh_reduce) = "act_reduce/tanh_reduce.cc";
    adf::runtime<ratio>(tanh_reduce) = 1;

    adf::connect<>(Wh_x.out[0], tanh_reduce.in[0]);
    adf::connect<>(Uh_h.out[0], tanh_reduce.in[1]);
    adf::connect<adf::parameter>(bh, tanh_reduce.in[2]);

    // R Gate Elements Output
    adf::connect<>(tanh_reduce.out[0], cand_h_output);
    // ----------------------------------------------------------------------

    }
}

#endif