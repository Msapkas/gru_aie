#include "gru.h"
#include <array>
#include "../config.h"

// Instantiate graph
gru gru_graph;

int main(int argc, char ** argv){

    // Array to simulate the RTPs
    std::array<float,H_VECTOR_SIZE> h_init_test_params;
    std::array<float,DIST_COEFF*X_VECTOR_SIZE> W_test_params;
    std::array<float,DIST_COEFF*H_VECTOR_SIZE> U_test_params;
    std::array<float,DIST_COEFF> b_test_params;

    for (int i = 0; i < H_VECTOR_SIZE; i++){h_init_test_params[i] = i;}
    for (int i = 0; i < DIST_COEFF*X_VECTOR_SIZE; i++){W_test_params[i] = i;}
    for (int i = 0; i < DIST_COEFF*H_VECTOR_SIZE; i++){U_test_params[i] = i;}
    for (int i = 0; i < DIST_COEFF; i++){b_test_params[i] = i;}

    gru_graph.init();

    // Pass all the RTPs
    for (int i = 0; i < H_VECTOR_SIZE/DIST_COEFF; i++) {
        gru_graph.update(gru_graph.hidden_initialization[i], h_init_test_params.data(), H_VECTOR_SIZE);
        gru_graph.update(gru_graph.W_params[i], W_test_params.data(), DIST_COEFF*X_VECTOR_SIZE);
        gru_graph.update(gru_graph.U_params[i], U_test_params.data(), DIST_COEFF*H_VECTOR_SIZE);
        gru_graph.update(gru_graph.b_params[i], b_test_params.data(), DIST_COEFF);
    }
    
    gru_graph.run(1);
    gru_graph.end();
    return 0;
};