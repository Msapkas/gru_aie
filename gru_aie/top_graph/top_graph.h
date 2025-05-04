#ifndef TOP_GRAPH_H
#define TOP_GRAPH_H

#include <adf.h>
#include "../GRU_cell/gru.h"
#include "../LINEAR_layers/linear_layers.h"

class top_graph : public adf::graph {
    public:

    // Pl IO
    adf::input_plio PL_INPUT;
    adf::output_plio PL_OUTPUT;

    // gru_under_top_graph
    gru gru_sub_graph;
    linear_layers linear_layer_sub_graph;

    top_graph () {

        // PL I/O
        PL_INPUT = adf::input_plio::create(adf::plio_32_bits, "aie_test_inputs/flat_x.txt");
        PL_OUTPUT = adf::output_plio::create(adf::plio_32_bits,"aie_y_pred.txt");

        adf::connect<adf::stream> (PL_INPUT.out[0], gru_sub_graph.GRU_INPUT);
        adf::connect<adf::stream> (gru_sub_graph.GRU_OUTPUT, linear_layer_sub_graph.LINEAR_INPUT);
        adf::connect<adf::stream> (linear_layer_sub_graph.LINEAR_OUTPUT, PL_OUTPUT.in[0]);

    }
};

#endif