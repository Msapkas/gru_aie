#ifndef GRU_h
#define GRU_h

#include "input_splitter.h"
#include "reset_and_candidate_gates.h"
#include "update_gate.h"
#include "new_hidden_state_gate.h"
#include "passthrough.h"

#define X_INPUT 24
#define HIDDEN 64

class GRU : public adf::graph {
public:

	adf::port<adf::input>  x_input;

	adf::kernel splitter;

	adf::kernel rst_n_cnd_gts;
	adf::port<adf::input> weights_Wxr;
	adf::port<adf::input> weights_Whr;
	adf::port<adf::input> biases_r;
	adf::port<adf::input> weights_Wxh;
	adf::port<adf::input> weights_Whh;
	adf::port<adf::input> biases_h;

	adf::kernel upd_gt;
	adf::port<adf::input> weights_Wxu;
	adf::port<adf::input> weights_Whu;
	adf::port<adf::input> biases_u;

	adf::kernel n_hdn_st_gt;

	adf::kernel feedback;

	adf::port<adf::output> h_output;

	GRU() {

		splitter = adf::kernel::create(input_splitter<X_INPUT>);
		adf::source(splitter) = "input_splitter.cc";
		adf::connect<adf::cascade> (x_input, splitter.in[0]);
		adf::runtime<adf::ratio>(splitter) = 1.;

		rst_n_cnd_gts = adf::kernel::create(reset_and_candidate_gates<X_INPUT,HIDDEN>);
		adf::source(rst_n_cnd_gts) = "reset_and_candidate_gates.cc";
		adf::connect<adf::cascade> (splitter.out[0], rst_n_cnd_gts.in[0]);

		adf::connect<adf::parameter> (weights_Whr, adf::async(rst_n_cnd_gts.in[2]));
		adf::connect<adf::parameter> (weights_Wxr, adf::async(rst_n_cnd_gts.in[3]));
		adf::connect<adf::parameter> (biases_r, adf::async(rst_n_cnd_gts.in[4]));
		adf::connect<adf::parameter> (weights_Wxh, adf::async(rst_n_cnd_gts.in[6]));
		adf::connect<adf::parameter> (weights_Whh, adf::async(rst_n_cnd_gts.in[5]));
		adf::connect<adf::parameter> (biases_h, adf::async(rst_n_cnd_gts.in[7]));

		adf::runtime<adf::ratio>(rst_n_cnd_gts) = 1.;

		upd_gt = adf::kernel::create(update_gate<X_INPUT,HIDDEN>);
		adf::source(upd_gt) = "update_gate.cc";
		adf::connect<adf::stream> (splitter.out[1], upd_gt.in[0]);

		adf::connect<adf::parameter> (weights_Whu, adf::async(upd_gt.in[2]));
		adf::connect<adf::parameter> (weights_Wxu, adf::async(upd_gt.in[3]));
		adf::connect<adf::parameter> (biases_u, adf::async(upd_gt.in[4]));

		adf::runtime<adf::ratio>(upd_gt) = 1.;

		n_hdn_st_gt = adf::kernel::create(new_hidden_state_gate<X_INPUT,HIDDEN>);
		adf::source(n_hdn_st_gt) = "new_hidden_state_gate.cc";
		adf::connect<adf::cascade> (rst_n_cnd_gts.out[0], n_hdn_st_gt.in[0]);
		adf::connect<adf::stream> (upd_gt.out[0], n_hdn_st_gt.in[1]);

		adf::runtime<adf::ratio>(n_hdn_st_gt) = 1.;

		feedback = adf::kernel::create(passthrough<HIDDEN>);
		adf::source(feedback) = "passthrough.cc";
		adf::connect<adf::stream> (n_hdn_st_gt.out[1], feedback.in[0]);
		adf::connect<adf::stream> (feedback.out[0], rst_n_cnd_gts.in[1]);
		adf::connect<adf::stream> (feedback.out[0], upd_gt.in[1]);

		adf::runtime<adf::ratio>(feedback) = 1.;

		adf::connect<adf::cascade> (n_hdn_st_gt.out[0],h_output);

	}

};

//PLIO plioIn ("plioIn" , plio_32_bits, "dataIn.txt" );
//PLIO plioOut("plioOut", plio_32_bits, "dataOut.txt");
//
//simulation::platform<1,1> plt(&plioIn, &plioOut);
//
//GRU gru;
//connect<> net0(plt.src[0], gru.x_input);
//connect<> net1(gru.h_output, plt.sink[0]);

#endif
