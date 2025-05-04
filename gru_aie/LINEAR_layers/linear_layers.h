#ifndef LINEAR_LAYERS
#define LINEAR_LAYERS

#include <adf.h>
#include "LINEAR_layers/FC_layer/fc_layer.h"
#include "LINEAR_layers/FC_layer/last_layer.h"
#include "../config.h"

class linear_layers: public adf::graph {
public:     
    adf::port<adf::input> LINEAR_INPUT;
    adf::port<adf::output> LINEAR_OUTPUT;

    adf::kernel first_layer;
    adf::port<adf::input> linear_params_0;
    adf::port<adf::input> bias_0;
    adf::port<adf::input> sequence_length_param;

    adf::kernel last_layer; 
    adf::port<adf::input> linear_params_1;
    adf::port<adf::input> bias_1;

    linear_layers () {

    first_layer = adf::kernel::create(fully_connected);
    adf::source(first_layer) = "LINEAR_layers/FC_layer/fc_layer.cc";
    adf::runtime<ratio>(first_layer) = 1;

    adf::connect<>(LINEAR_INPUT, first_layer.in[0]);
    adf::connect<adf::parameter>(linear_params_0, adf::async(first_layer.in[1]));
    adf::connect<adf::parameter>(bias_0, adf::async(first_layer.in[2]));
    // adf::connect<adf::parameter>(sequence_length_param, adf::async(first_layer.in[3]));

    last_layer = adf::kernel::create(last_fully_connected);
    adf::source(last_layer) = "LINEAR_layers/FC_layer/last_layer.cc";
    adf::runtime<ratio>(last_layer) = 1;

    adf::connect<>(first_layer.out[0], last_layer.in[0]);
    adf::connect<adf::parameter>(linear_params_1, adf::async(last_layer.in[1]));
    adf::connect<adf::parameter>(bias_1, adf::async(last_layer.in[2]));

    adf::connect<>(last_layer.out[0], LINEAR_OUTPUT);
    }

};

#endif