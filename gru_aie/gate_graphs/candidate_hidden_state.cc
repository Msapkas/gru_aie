#include "candidate_hidden_state.h"
#include "../config.h"

candidate_hidden_gate candidate_hidden_gate;

int main(int argc, char ** argv)
{   
    candidate_hidden_gate.init();
    candidate_hidden_gate.run(-1);
    candidate_hidden_gate.end();
    return 0;
}