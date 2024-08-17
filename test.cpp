#include <iostream>
#include <vector>
#include <string>
#include "cnpy.h"

int main(int argc, char* argv[]) {
    std::vector<double> gt = {0, 0};
    std::vector<int> img_size = {720, 1280};

    SosObjective obj;

    for (int keyname = 1; keyname < 2; ++keyname) {
        std::string basedir = "H:\\propeller_det_data\\testbed_exp\\data\\event\\processed\\diff_illumination\\" + std::to_string(keyname);
        std::string savedir = "test.csv";

        std::cout << "Processing " << keyname << "..." << std::endl;

        double blur = 0.0;
        std::vector<double> centers = {600, 411};

        std::vector<std::vector<double>> rpms_res;
        std::string filepath = basedir + "\\cluster_1.npy";

        // Load data from file
        cnpy::NpyArray arr = cnpy::npy_load(filepath);
        double* loaded_data = arr.data<double>();
        std::vector<std::vector<double>> alldata(arr.shape[0], std::vector<double>(arr.shape[1]));

        for (size_t i = 0; i < arr.shape[0]; ++i) {
            for (size_t j = 0; j < arr.shape[1]; ++j) {
                alldata[i][j] = loaded_data[i * arr.shape[1] + j];
            }
        }

        for (size_t idx = 0; idx < alldata.size(); ++idx) {
            auto data = alldata[idx];
            data.resize(10);

            auto begin = std::chrono::high_resolution_clock::now();
            auto [rpm, rad_s] = calculate_rpm(data, img_size, obj, blur, centers);
            auto end = std::chrono::high_resolution_clock::now();

            std::cout << " " << idx << "/" << alldata.size() << ":  rpm = " << std::abs(rpm) << std::endl;

            double timestamp = idx * 50e3;
            double delta_t = std::chrono::duration<double>(end - begin).count();
            std::cout << "Latency: " << delta_t << "s" << std::endl;
            rpms_res.push_back({timestamp, std::abs(rpm), delta_t});
        }

        std::ofstream file(savedir);
        file << "Time,RPM,DeltaT\n";
        for (const auto& row : rpms_res) {
            file << row[0] << "," << row[1] << "," << row[2] << "\n";
        }
        file.close();
    }

    return 0;
}