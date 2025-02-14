#ifndef GRU_ATOM_H
#define GRU_ATOM_H

#include <adf.h>
#include "mat_vec_mul/mat_input_vec_mul.h"
#include "mat_vec_mul/mat_hidden_vec_mul.h"
#include "act_reduce/sigmoid_reduce.h"
#include "act_reduce/tanh_reduce.h"
#include "new_hidden_state.h"
#include "config.h"

class gru_atom: public adf::graph {
public:
    // PL Connections - IO
    adf::input_plio plio_in;
    adf::output_plio plio_out;
    // ------------------------------
    // Graph Inputs
    adf::port<adf::input> x_input;
    adf::port<adf::input> h_init;
    // ------------------------------
    adf::kernel Wr_x;
    adf::port<adf::input> Wr;
    adf::kernel Ur_h;
    adf::port<adf::input> Ur;
    adf::kernel r_sigm_reduce;
    adf::port<adf::input> br;
    // ------------------------------
    adf::kernel Wz_x;
    adf::port<adf::input> Wz;
    adf::kernel Uz_h;
    adf::port<adf::input> Uz;
    adf::kernel z_sigm_reduce;
    adf::port<adf::input> bz;
    // ------------------------------

    adf::kernel Wh_x;
    adf::port<adf::input> Wh;
    adf::kernel Uz_hr;
    adf::port<adf::input> Uh;
    adf::kernel hhat_tanh_reduce;
    adf::port<adf::input> bh;
    // ------------------------------
    adf::kernel final_hidden_state;
    // ------------------------------
    adf::port<adf::output> hidden_state_out;
    // ------------------------------

    gru_atom() {

            plio_in = adf::input_plio::create(adf::plio_128_bits, "data/test_in.txt");
            plio_out = adf::output_plio::create(adf::plio_128_bits, "data/test_out.txt");

            // Reset Gate (R) -------------------------------------------------------

            // Wr[N](rows) x x_input_vector
            Wr_x = adf::kernel::create(mat_input_vec_mul);
            adf::source(Wr_x) = "mat_vec_mul/mat_input_vec_mul.cc";
            adf::runtime<ratio>(Wr_x) = 1;

            adf::connect<>(x_input, Wr_x.in[0]);
            adf::connect<adf::parameter>(Wr, adf::async(Wr_x.in[1]));

            // Ur[N](rows) x prev_hidden_vector
            Ur_h = adf::kernel::create(mat_hidden_vec_mul);
            adf::source(Ur_h) = "mat_vec_mul/mat_hidden_vec_mul.cc";
            adf::runtime<ratio>(Ur_h) = 1;

            adf::connect<adf::parameter>(Ur, adf::async(Ur_h.in[1]));
            adf::connect<adf::parameter>(h_init, adf::async(Ur_h.in[2]));

            // Reduce_add and apply sigmoid LUT
            r_sigm_reduce = adf::kernel::create(sigmoid_reduce);
            adf::source(r_sigm_reduce) = "act_reduce/sigmoid_reduce.cc";
            adf::runtime<ratio>(r_sigm_reduce) = 1;

            adf::connect<>(Wr_x.out[0], r_sigm_reduce.in[0]);
            adf::connect<>(Ur_h.out[0], r_sigm_reduce.in[1]);
            adf::connect<adf::parameter>(br, r_sigm_reduce.in[2]);

            // ----------------------------------------------------------------------

            // Update (Z) Gate ------------------------------------------------------

            // Wz[N](rows) x x_input_vector
            Wz_x = adf::kernel::create(mat_input_vec_mul);
            adf::source(Wz_x) = "mat_vec_mul/mat_input_vec_mul.cc";
            adf::runtime<ratio>(Wz_x) = 1;

            adf::connect<>(x_input, Wz_x.in[0]);
            adf::connect<adf::parameter>(Wz, adf::async(Wz_x.in[1]));

            // Uz[N](rows) x prev_hidden_vector
            Uz_h = adf::kernel::create(mat_hidden_vec_mul);
            adf::source(Uz_h) = "mat_vec_mul/mat_hidden_vec_mul.cc";
            adf::runtime<ratio>(Uz_h) = 1;

            adf::connect<adf::parameter>(Ur, adf::async(Uz_h.in[1]));
            adf::connect<adf::parameter>(h_init, adf::async(Uz_h.in[2]));

            // Reduce_add and apply sigmoid LUT
            z_sigm_reduce = adf::kernel::create(sigmoid_reduce);
            adf::source(r_sigm_reduce) = "act_reduce/sigmoid_reduce.cc";
            adf::runtime<ratio>(r_sigm_reduce) = 1;

            adf::connect<>(Wz_x.out[0], z_sigm_reduce.in[0]);
            adf::connect<>(Uz_h.out[0], z_sigm_reduce.in[1]);
            adf::connect<adf::parameter>(br, z_sigm_reduce.in[2]);

            // -----------------------------------------------------------------------

            // Candidate Hidden State (H_hat) Gate  ----------------------------------

            // Wh[N](rows) x x_input_vector
            Wh_x = adf::kernel::create(mat_input_vec_mul);
            adf::source(Wh_x) = "mat_vec_mul/mat_input_vec_mul.cc";
            adf::runtime<ratio>(Wh_x) = 1;

            adf::connect<>(x_input, Wh_x.in[0]);
            adf::connect<adf::parameter>(Wh, adf::async(Wh_x.in[1]));

            // Uh[N](rows) x ( prev_hidden_vector dot reset gate ) 
            Uz_hr = adf::kernel::create(mat_hidden_vec_mul);
            adf::source(Uz_h) = "mat_vec_mul/mat_hidden_vec_mul.cc";
            adf::runtime<ratio>(Uz_hr) = 1;

            adf::connect<adf::parameter>(Uh, adf::async(Uz_hr.in[1]));
            adf::connect<adf::parameter>(h_init, adf::async(Uz_hr.in[2]));

            // Reduce_add and apply tanh LUT
            hhat_tanh_reduce = adf::kernel::create(tanh_reduce);
            adf::source(hhat_tanh_reduce) = "act_reduce/tanh_reduce.cc";
            adf::runtime<ratio>(hhat_tanh_reduce) = 1;

            adf::connect<>(Wh_x.out[0], hhat_tanh_reduce.in[0]);
            adf::connect<>(Uz_hr.out[0], hhat_tanh_reduce.in[1]);
            adf::connect<adf::parameter>(bh, hhat_tanh_reduce.in[2]);

            // ----------------------------------------------------------------------

            // New Hidden State (H) -------------------------------------------------

            final_hidden_state = adf::kernel::create(new_hidden_state);
            adf::source(final_hidden_state) = "new_hidden_state.cc";
            adf::runtime<ratio>(final_hidden_state) = 1;

            adf::connect<>(hhat_tanh_reduce.out[0], final_hidden_state.in[0]); 
            adf::connect<>(z_sigm_reduce.out[0], final_hidden_state.in[1]);

            // Output of the model
            adf::connect<>(final_hidden_state.out[0], hidden_state_out);

            // Pass new hidden state to relevant kernels (feedback)
            adf::connect<>(final_hidden_state.out[0], Ur_h.in[0]);
            adf::connect<>(final_hidden_state.out[0], Uz_h.in[0]);
            adf::connect<>(final_hidden_state.out[0], Uz_hr.in[0]);
            
            // ----------------------------------------------------------------------
            

            // ----------------------------------------------------------------------
        }
};

#endif
