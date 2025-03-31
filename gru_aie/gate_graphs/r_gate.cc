#include "r_gate.h"
#include "../config.h"

r_gate r_gate;

int main(int argc, char ** argv)
{   
    r_gate.init();
    r_gate.run();
    r_gate.end();
    return 0;
}