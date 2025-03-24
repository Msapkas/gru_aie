#include "r_gate.h"
#include "../config.h"

r_gate r_gate;

int main(int argc, char ** argv)
{   
    r_gate.init();
    r_gate.run(-1);
    r_gate.end();
    return 0;
}