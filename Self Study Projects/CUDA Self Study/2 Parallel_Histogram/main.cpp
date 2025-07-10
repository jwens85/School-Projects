//
// Created by jwens on 7/10/2025.
//
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

void compute_histogram_gpu(const std::vector<int>& pixels, std::vector<int>& histogram);

std::vector<int> read_pixel_data(const std::string& filename, int max_rows = 1000) {
    std::vector<int> pixels;
    std::ifstream file(filename);
    std::string line;

    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return pixels;
    }

    // Skip header
    std::getline(file, line);

    int row_count = 0;
    while (std::getline(file, line) && row_count < max_rows) {
        std::stringstream ss(line);
        std::string cell;
        int column_index = 0;

        // If this is the digit recognizer dataset, first column is the label
        while (std::getline(ss, cell, ',')) {
            if (column_index > 0) {
                pixels.push_back(std::stoi(cell));
            }
            ++column_index;
        }

        ++row_count;
    }

    file.close();
    return pixels;
}

int main() {
    std::string filename = "data/train.csv";  // Adjust path as needed
    std::vector<int> pixel_values = read_pixel_data(filename);

    if (pixel_values.empty()) {
        std::cerr << "No pixel data loaded. Exiting." << std::endl;
        return 1;
    }

    std::vector<int> histogram(256, 0);  // 256 grayscale bins

    compute_histogram_gpu(pixel_values, histogram);

    std::cout << "Histogram result (first 10 bins):" << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << "Bin " << i << ": " << histogram[i] << std::endl;
    }

    return 0;
}
