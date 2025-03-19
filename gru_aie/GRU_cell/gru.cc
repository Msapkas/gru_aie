#include "gru.h"
#include <array>
#include "../config.h"

// Instantiate graph
gru gru_graph;

int main(int argc, char ** argv){

    gru_graph.init();

    // Array to simulate the RTPs
    std::array<float,H_VECTOR_SIZE> h_init_test_params;
    std::array<unsigned int,NKERNELS> ID_param;
    std::array<float,DIST_COEFF*X_VECTOR_SIZE> W_test_params;
    std::array<float,DIST_COEFF*H_VECTOR_SIZE> U_test_params;
    std::array<float,DIST_COEFF> b_test_params;

    for (int i = 0; i < H_VECTOR_SIZE; i++){h_init_test_params[i] = 1;}
    for (unsigned int i = 0; i < NKERNELS; i++){ID_param[i] = i*DIST_COEFF;}
    for (int i = 0; i < DIST_COEFF*X_VECTOR_SIZE; i++){W_test_params[i] = 1;}
    for (int i = 0; i < DIST_COEFF*H_VECTOR_SIZE; i++){U_test_params[i] = 1;}
    for (int i = 0; i < DIST_COEFF; i++){b_test_params[i] = 0;}

    // Pass all the RTPs
    for (int i = 0; i < NKERNELS; i++) {

        gru_graph.update(gru_graph.r_identifier[i], &ID_param[i], 1);
        gru_graph.update(gru_graph.r_hidden_initialization[i], h_init_test_params.data(), H_VECTOR_SIZE);
        gru_graph.update(gru_graph.Wr_params[i], W_test_params.data(), DIST_COEFF*X_VECTOR_SIZE);
        gru_graph.update(gru_graph.Ur_params[i], U_test_params.data(), DIST_COEFF*H_VECTOR_SIZE);
        gru_graph.update(gru_graph.br_params[i], b_test_params.data(), DIST_COEFF);

        gru_graph.update(gru_graph.z_identifier[i], &ID_param[i], 1);
        gru_graph.update(gru_graph.z_hidden_initialization[i], h_init_test_params.data(), H_VECTOR_SIZE);
        gru_graph.update(gru_graph.Wz_params[i], W_test_params.data(), DIST_COEFF*X_VECTOR_SIZE);
        gru_graph.update(gru_graph.Uz_params[i], U_test_params.data(), DIST_COEFF*H_VECTOR_SIZE);
        gru_graph.update(gru_graph.bz_params[i], b_test_params.data(), DIST_COEFF);

        gru_graph.update(gru_graph.chsg_identifier[i], &ID_param[i], 1);
        gru_graph.update(gru_graph.chsg_hidden_initialization[i], h_init_test_params.data(), H_VECTOR_SIZE);
        gru_graph.update(gru_graph.Wh_params[i], W_test_params.data(), DIST_COEFF*X_VECTOR_SIZE);
        gru_graph.update(gru_graph.Uh_params[i], U_test_params.data(), DIST_COEFF*H_VECTOR_SIZE);
        gru_graph.update(gru_graph.bh_params[i], b_test_params.data(), DIST_COEFF);

    }
    gru_graph.update(gru_graph.new_hidden_state_gate_hidden_initialization, h_init_test_params.data(), H_VECTOR_SIZE);
    //

    gru_graph.wait(10);
    gru_graph.run(1);
    gru_graph.end();
    return 0;

};