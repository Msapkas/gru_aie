#include "GRU.h"
#include <fstream>
#include <array>

using namespace adf;

#if defined(__X86SIM__) || defined(__ADF_FRONTEND__) || defined(__AIESIM__)

int main(int argc, char** argv) {
	gru.init();
	gru.run(1);
	gru.end();
	return 0;
}
