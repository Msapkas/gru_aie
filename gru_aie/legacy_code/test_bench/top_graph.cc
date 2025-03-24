#include "top_graph.h"

// Instantiate graph
test_bench test_bench;

int main(int argc, char ** argv){

    test_bench.init();
    //test_bench.wait(10);
    test_bench.run(1);
    test_bench.end();
    return 0;
}