#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iomanip>

std::vector<std::vector<double>> read_txt_xypt(const std::string& file_path, int max_rows = -1) {
    std::ifstream file(file_path);
    std::vector<std::vector<double>> data;

    if (!file.is_open()) {
        std::cerr << "Failed to open the file." << std::endl;
        return data;
    }

    std::string line;
    int row_count = 0;
    while (std::getline(file, line) && (max_rows == -1 || row_count < max_rows)) {
        std::istringstream iss(line);
        std::vector<double> row;
        double value;
        while (iss >> value) {
            row.push_back(value);
        }
        data.push_back(row);
        row_count++;
    }
    file.close();

    std::vector<double> x_raw, y_raw, p_raw, time_raw;
    for (const auto& row : data) {
        if (row.size() >= 4) {
            x_raw.push_back(row[0]);
            y_raw.push_back(row[1]);
            p_raw.push_back(row[2]);
            time_raw.push_back(row[3]);
        }
    }
    
    std::vector<std::vector<double>> new_data(x_raw.size(), std::vector<double>(3));
    for (size_t i = 0; i < x_raw.size(); ++i) {
        new_data[i][0] = x_raw[i];
        new_data[i][1] = y_raw[i];
        new_data[i][2] = time_raw[i];
    }

    std::cout << "Original data shape: (" << data.size() << ", " << (data.empty() ? 0 : data[0].size()) << ")" << std::endl;
    std::cout << "New data shape: (" << new_data.size() << ", 3)" << std::endl;

    return new_data;
}

int main() {
    int img_height = 720;
    int img_width = 1280;

    int keyname_start = 1;
    int keyname_end = 2;
    int keyname_step = 1;

    for (int keyname = keyname_start; keyname < keyname_end; keyname += keyname_step) {
        std::string basedir = "/home/goolo/projects/fast_propeller/";
        std::string savedir = "test.csv";

        std::cout << "Processing " << keyname << "..." << std::endl;

        double blur = 0.0;
        std::vector<int> centers = {600, 411};

        std::vector<std::vector<double>> rpms_res;
        std::string filepath = basedir + "1_sub.txt";
        std::vector<std::vector<double>> alldata = read_txt_xypt(filepath);
    }

    return 0;
}