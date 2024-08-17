#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <torch/script.h>

class ObjectiveFunction {
public:
    std::string name;
    bool use_polarity;
    bool has_derivative;
    double default_blur;

    ObjectiveFunction(std::string name = "template", bool use_polarity = true, bool has_derivative = true, double default_blur = 1.0)
        : name(name), use_polarity(use_polarity), has_derivative(has_derivative), default_blur(default_blur) {}
};

class SosObjective : public ObjectiveFunction {
public:
    SosObjective() : ObjectiveFunction("sos", true, true, 1.0) {}

    double evaluate_function(const std::vector<double>& params, const std::vector<double>& xs, const std::vector<double>& ys, const std::vector<double>& ts,
                             std::function<void()> warpfunc, const std::vector<int>& img_size, double blur_sigma, bool showimg, cv::Mat& iwe) {
        if (iwe.empty()) {
            // get_iwe function needs to be implemented
        }
        blur_sigma = (blur_sigma == 0) ? default_blur : blur_sigma;
        if (blur_sigma > 0) {
            cv::GaussianBlur(iwe, iwe, cv::Size(0, 0), blur_sigma);
        }
        cv::Mat iwe_squared;
        cv::multiply(iwe, iwe, iwe_squared);
        double sos = cv::mean(iwe_squared)[0];
        return -sos;
    }
};

std::vector<double> events_bounds_mask(const std::vector<double>& xs, const std::vector<double>& ys, double x_min, double x_max, double y_min, double y_max) {
    std::vector<double> mask(xs.size(), 1.0);
    for (size_t i = 0; i < xs.size(); ++i) {
        if (xs[i] <= x_min || xs[i] > x_max || ys[i] <= y_min || ys[i] > y_max) {
            mask[i] = 0.0;
        }
    }
    return mask;
}

void interpolate_to_image(const std::vector<int>& pxs, const std::vector<int>& pys, const std::vector<double>& dxs, const std::vector<double>& dys,
                          const std::vector<double>& weights, cv::Mat& img) {
    for (size_t i = 0; i < pxs.size(); ++i) {
        img.at<double>(pys[i], pxs[i]) += weights[i] * (1.0 - dxs[i]) * (1.0 - dys[i]);
        img.at<double>(pys[i], pxs[i] + 1) += weights[i] * dxs[i] * (1.0 - dys[i]);
        img.at<double>(pys[i] + 1, pxs[i]) += weights[i] * (1.0 - dxs[i]) * dys[i];
        img.at<double>(pys[i] + 1, pxs[i] + 1) += weights[i] * dxs[i] * dys[i];
    }
}

void interpolate_to_derivative_img(const std::vector<int>& pxs, const std::vector<int>& pys, const std::vector<double>& dxs, const std::vector<double>& dys,
                                   std::vector<cv::Mat>& d_img, const std::vector<std::vector<double>>& w1, const std::vector<std::vector<double>>& w2) {
    for (size_t i = 0; i < d_img.size(); ++i) {
        for (size_t j = 0; j < pxs.size(); ++j) {
            d_img[i].at<double>(pys[j], pxs[j]) += w1[i][j] * (-(1.0 - dys[j])) + w2[i][j] * (-(1.0 - dxs[j]));
            d_img[i].at<double>(pys[j], pxs[j] + 1) += w1[i][j] * (1.0 - dys[j]) + w2[i][j] * (-dxs[j]);
            d_img[i].at<double>(pys[j] + 1, pxs[j]) += w1[i][j] * (-dys[j]) + w2[i][j] * (1.0 - dxs[j]);
            d_img[i].at<double>(pys[j] + 1, pxs[j] + 1) += w1[i][j] * dys[j] + w2[i][j] * dxs[j];
        }
    }
}

std::pair<cv::Mat, std::vector<cv::Mat>> events_to_image_drv(const std::vector<double>& xn, const std::vector<double>& yn, const std::vector<double>& pn,
                                                             const std::vector<std::vector<double>>& jacobian_xn, const std::vector<std::vector<double>>& jacobian_yn,
                                                             const std::vector<int>& sensor_size, bool clip_out_of_range, bool interpolation, bool padding, bool compute_gradient) {
    std::vector<double> xs = xn;
    std::vector<double> ys = yn;
    std::vector<double> ps = pn;

    std::vector<std::vector<double>> jacobian_x, jacobian_y;
    if (compute_gradient) {
        jacobian_x = jacobian_xn;
        jacobian_y = jacobian_yn;
    }

    std::vector<double> mask(xs.size(), 1.0);
    if (clip_out_of_range) {
        double clipx = (interpolation == false && padding == false) ? sensor_size[1] : sensor_size[1] - 1;
        double clipy = (interpolation == false && padding == false) ? sensor_size[0] : sensor_size[0] - 1;
        for (size_t i = 0; i < xs.size(); ++i) {
            if (xs[i] >= clipx || ys[i] >= clipy) {
                mask[i] = 0.0;
            }
        }
    }

    std::vector<int> pxs(xs.size()), pys(ys.size());
    std::vector<double> dxs(xs.size()), dys(ys.size());
    for (size_t i = 0; i < xs.size(); ++i) {
        pxs[i] = static_cast<int>(std::floor(xs[i]));
        pys[i] = static_cast<int>(std::floor(ys[i]));
        dxs[i] = xs[i] - pxs[i];
        dys[i] = ys[i] - pys[i];
        pxs[i] = static_cast<int>(pxs[i] * mask[i]);
        pys[i] = static_cast<int>(pys[i] * mask[i]);
        ps[i] = ps[i] * mask[i];
    }

    cv::Mat img(sensor_size[0], sensor_size[1], CV_64F, cv::Scalar(0));
    interpolate_to_image(pxs, pys, dxs, dys, ps, img);

    std::vector<cv::Mat> d_img;
    if (compute_gradient) {
        d_img.resize(2, cv::Mat(sensor_size[0], sensor_size[1], CV_64F, cv::Scalar(0)));
        std::vector<std::vector<double>> w1(2, std::vector<double>(xs.size())), w2(2, std::vector<double>(xs.size()));
        for (size_t i = 0; i < xs.size(); ++i) {
            w1[0][i] = jacobian_x[0][i] * ps[i];
            w1[1][i] = jacobian_x[1][i] * ps[i];
            w2[0][i] = jacobian_y[0][i] * ps[i];
            w2[1][i] = jacobian_y[1][i] * ps[i];
        }
        interpolate_to_derivative_img(pxs, pys, dxs, dys, d_img, w1, w2);
    }
    return {img, d_img};
}

std::pair<cv::Mat, std::vector<cv::Mat>> get_iwe(const std::vector<double>& params, const std::vector<double>& xs, const std::vector<double>& ys, const std::vector<double>& ts,
                                                 std::function<void()> warpfunc, const std::vector<int>& img_size, bool compute_gradient, bool use_polarity) {
    // warpfunc needs to be implemented
    std::vector<double> jx, jy;
    std::vector<double> mask = events_bounds_mask(xs, ys, 0, img_size[1], 0, img_size[0]);
    std::vector<double> xs_masked(xs.size()), ys_masked(ys.size());
    for (size_t i = 0; i < xs.size(); ++i) {
        xs_masked[i] = xs[i] * mask[i];
        ys_masked[i] = ys[i] * mask[i];
    }

    std::vector<double> ps(xs.size(), 1.0);
    auto [iwe, iwe_drv] = events_to_image_drv(xs_masked, ys_masked, ps, jx, jy, img_size, true, true, true, compute_gradient);
    return {iwe, iwe_drv};
}

class WarpFunction {
public:
    std::string name;
    int dims;

    WarpFunction(std::string name, int dims) : name(name), dims(dims) {}
};

class RotvelWarp : public WarpFunction {
public:
    std::vector<double> centers;

    RotvelWarp(const std::vector<double>& centers = {}) : WarpFunction("rotvel_warp", 1), centers(centers) {}

    std::tuple<std::vector<double>, std::vector<double>, std::vector<std::vector<double>>, std::vector<std::vector<double>>>
    warp(const std::vector<double>& xs, const std::vector<double>& ys, const std::vector<double>& ts, double t0, const std::vector<double>& params, bool compute_grad) {
        std::vector<double> dt(ts.size());
        for (size_t i = 0; i < ts.size(); ++i) {
            dt[i] = ts[i] - t0;
        }
        std::vector<double> theta(ts.size());
        for (size_t i = 0; i < ts.size(); ++i) {
            theta[i] = dt[i] * params[0];
        }

        std::vector<double> xs_copy = xs;
        std::vector<double> ys_copy = ys;

        double xs_mean = (centers.empty()) ? std::accumulate(xs_copy.begin(), xs_copy.end(), 0.0) / xs_copy.size() : centers[0];
        double ys_mean = (centers.empty()) ? std::accumulate(ys_copy.begin(), ys_copy.end(), 0.0) / ys_copy.size() : centers[1];

        for (size_t i = 0; i < xs_copy.size(); ++i) {
            xs_copy[i] -= xs_mean;
            ys_copy[i] -= ys_mean;
        }

        std::vector<double> cos_theta(theta.size()), sin_theta(theta.size());
        for (size_t i = 0; i < theta.size(); ++i) {
            cos_theta[i] = std::cos(theta[i]);
            sin_theta[i] = std::sin(theta[i]);
        }

        std::vector<double> xs_tmp(xs_copy.size()), ys_tmp(ys_copy.size());
        for (size_t i = 0; i < xs_copy.size(); ++i) {
            xs_tmp[i] = xs_copy[i] * cos_theta[i] - ys_copy[i] * sin_theta[i];
            ys_tmp[i] = xs_copy[i] * sin_theta[i] + ys_copy[i] * cos_theta[i];
        }

        std::vector<double> x_prime(xs_tmp.size()), y_prime(ys_tmp.size());
        for (size_t i = 0; i < xs_tmp.size(); ++i) {
            x_prime[i] = xs_tmp[i] + xs_mean;
            y_prime[i] = ys_tmp[i] + ys_mean;
        }

        std::vector<std::vector<double>> jacobian_x, jacobian_y;
        if (compute_grad) {
            jacobian_x.resize(2, std::vector<double>(x_prime.size()));
            jacobian_y.resize(2, std::vector<double>(y_prime.size()));
            for (size_t i = 0; i < x_prime.size(); ++i) {
                jacobian_x[0][i] = -ys[i] * sin_theta[i];
                jacobian_x[1][i] = xs[i] * sin_theta[i];
                jacobian_y[0][i] = xs[i] * cos_theta[i];
                jacobian_y[1][i] = ys[i] * cos_theta[i];
            }
        }

        return {x_prime, y_prime, jacobian_x, jacobian_y};
    }
};

std::vector<double> optimize(const std::vector<double>& xs, const std::vector<double>& ys, const std::vector<double>& ts, RotvelWarp& warp_function,
                             SosObjective& objective, const std::vector<int>& img_size, double blur_sigma) {
    std::vector<double> x0 = {-std::deg2rad(-18000)};
    std::vector<double> bounds = {-std::deg2rad(20000), -std::deg2rad(7000)};
    std::cout << bounds[0] << ", " << bounds[1] << std::endl;

    if (x0.empty()) {
        x0.resize(warp_function.dims, 0.0);
    }

    // Optimization logic needs to be implemented
    std::vector<double> argmax = {0.0}; // Placeholder

    std::cout << argmax[0] << std::endl;
    return argmax;
}

std::tuple<std::vector<double>, std::vector<double>, std::vector<double>> get_data(const std::vector<std::vector<double>>& data) {
    std::vector<double> xs(data.size()), ys(data.size()), ts(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        xs[i] = static_cast<int>(data[i][0]);
        ys[i] = static_cast<int>(data[i][1]);
        ts[i] = static_cast<int>(data[i][2]);
    }

    double ts0 = ts[0];
    for (size_t i = 0; i < ts.size(); ++i) {
        ts[i] = (ts[i] - ts0) / 1e6;
    }

    std::cout << xs.size() << std::endl;
    return {xs, ys, ts};
}

std::pair<double, std::vector<double>> calculate_rpm(const std::vector<std::vector<double>>& data, const std::vector<int>& img_size, SosObjective& obj, double blur, const std::vector<double>& centers) {
    auto [xs, ys, ts] = get_data(data);

    RotvelWarp warp(centers);
    std::vector<double> argmax = optimize(xs, ys, ts, warp, obj, img_size, blur);

    double rpm = std::rad2deg(argmax[0]) / 360 * 60;
    return {rpm, argmax};
}

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
        // Load data from file (needs to be implemented)
        std::vector<std::vector<double>> alldata; // Placeholder

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