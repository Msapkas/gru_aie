#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <fstream>

// Load floats from file into numbers array and print status
inline void loadFloatsFromFile(const char* filename, float numbers[], int maxSize) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << '\n';
        return;
    }

    float value;
    int count = 0;

    for (int i = 0; i < maxSize; ++i) {
        if (file >> value) {
            numbers[i] = value;
            ++count;
        } else {
            break;
        }
    }

    file.close();

    std::cout << "Passed " << count << " float(s) from " << filename << " successfully.\n";
}

#endif 