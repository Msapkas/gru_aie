#ifndef TEST_BENCH_H
#define TEST_BENCH_H

#include <adf.h>
#include "GRU_cell/gru.h"

class test_bench : public adf::graph {
    public:

    // Pl IO
    adf::input_plio PL_INPUT;
    adf::output_plio PL_OUTPUT;

    // gru_under_test
    gru gru;

    test_bench () {

        // PL I/O
        PL_INPUT = adf::input_plio::create(adf::plio_128_bits, "data/test_in_x.txt");
        PL_OUTPUT = adf::output_plio::create(adf::plio_128_bits,"data/outputs.txt");

        adf::connect<adf::stream> (PL_INPUT, gru.PL_INPUT);
        adf::connect<adf::stream> (gru.PL_OUTPUT, PL_OUTPUT.in[0]);

    }
};

#endif