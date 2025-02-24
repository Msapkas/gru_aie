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
    for (int i = 0; i < DIST_COEFF; i++){b_test_params[i] = 1;}

    gru_graph.init();
    gru_graph.run(1);

    // Pass all the RTPs
    for (int i = 0; i < H_VECTOR_SIZE/DIST_COEFF; i++) {

        gru_graph.update(gru_graph.r_hidden_initialization[i], h_init_test_params.data(), H_VECTOR_SIZE);
        gru_graph.update(gru_graph.Wr_params[i], W_test_params.data(), DIST_COEFF*X_VECTOR_SIZE);
        gru_graph.update(gru_graph.Ur_params[i], U_test_params.data(), DIST_COEFF*H_VECTOR_SIZE);
        gru_graph.update(gru_graph.br_params[i], b_test_params.data(), DIST_COEFF);

        // gru_graph.update(gru_graph.z_hidden_initialization[i], h_init_test_params.data(), H_VECTOR_SIZE);
        // gru_graph.update(gru_graph.Wz_params[i], W_test_params.data(), DIST_COEFF*X_VECTOR_SIZE);
        // gru_graph.update(gru_graph.Uz_params[i], U_test_params.data(), DIST_COEFF*H_VECTOR_SIZE);
        // gru_graph.update(gru_graph.bz_params[i], b_test_params.data(), DIST_COEFF);
        
    }
    
    gru_graph.end();
    return 0;
};