#include "top_graph.h"

// Instantiate graph
top_graph top_graph;

int main(int argc, char ** argv){

    top_graph.init();
    //test_bench.wait(10);
    top_graph.run(1);
    top_graph.end();
    return 0;
}