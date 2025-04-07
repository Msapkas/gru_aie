#include "z_gate.h"
#include "../config.h"

z_gate z_gate;

int main(int argc, char ** argv)
{   
    z_gate.init();
    z_gate.run(-1);
    z_gate.end();
    return 0;
}