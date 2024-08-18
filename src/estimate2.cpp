#include <iostream>
#include <vector>
#include <cmath>
#include <ceres/ceres.h>
#include <Eigen/Dense>
#include <fstream>
#include <sstream>
#include <chrono>

using namespace std;
using namespace Eigen;

class ObjectiveFunction {
public:
    ObjectiveFunction(string name = "template", bool use_polarity = true, bool has_derivative = true, double default_blur = 1.0)
        : name(name), use_polarity(use_polarity), has_derivative(has_derivative), default_blur(default_blur) {}

    virtual ~ObjectiveFunction() = default;

    virtual double evaluate_function(const VectorXd& params, const vector<int>& xs, const vector<int>& ys, const vector<double>& ts,
                                     const function<void(const vector<int>&, const vector<int>&, const vector<double>&, const VectorXd&, vector<int>&, vector<int>&)>& warpfunc,
                                     const Vector2i& img_size, double blur_sigma, bool showimg, MatrixXd& iwe) = 0;

protected:
    string name;
    bool use_polarity;
    bool has_derivative;
    double default_blur;
};

class SosObjective : public ObjectiveFunction {
public:
    SosObjective() : ObjectiveFunction("sos", true, true, 1.0) {}

    double evaluate_function(const VectorXd& params, const vector<int>& xs, const vector<int>& ys, const vector<double>& ts,
                             const function<void(const vector<int>&, const vector<int>&, const vector<double>&, const VectorXd&, vector<int>&, vector<int>&)>& warpfunc,
                             const Vector2i& img_size, double blur_sigma, bool showimg, MatrixXd& iwe) override {
        if (iwe.size() == 0) {
            get_iwe(params, xs, ys, ts, warpfunc, img_size, false, iwe);
        }
        blur_sigma = default_blur;
        double sos = (iwe.array() * iwe.array()).mean();
        return -sos;
    }

private:
    void get_iwe(const VectorXd& params, const vector<int>& xs, const vector<int>& ys, const vector<double>& ts,
                 const function<void(const vector<int>&, const vector<int>&, const vector<double>&, const VectorXd&, vector<int>&, vector<int>&)>& warpfunc,
                 const Vector2i& img_size, bool compute_gradient, MatrixXd& iwe) {
        vector<int> warped_xs, warped_ys;
        warpfunc(xs, ys, ts, params, warped_xs, warped_ys);
        iwe = events_to_image_drv(warped_xs, warped_ys, img_size);
    }

    MatrixXd events_to_image_drv(const vector<int>& xs, const vector<int>& ys, const Vector2i& img_size) {
        MatrixXd img = MatrixXd::Zero(img_size(0), img_size(1));
        for (size_t i = 0; i < xs.size(); ++i) {
            if (xs[i] >= 0 && xs[i] < img_size(1) && ys[i] >= 0 && ys[i] < img_size(0)) {
                img(ys[i], xs[i]) += 1.0;
            }
        }
        return img;
    }
};

class WarpFunction {
public:
    WarpFunction(string name, int dims) : name(name), dims(dims) {}

    virtual ~WarpFunction() = default;

    virtual void warp(const vector<int>& xs, const vector<int>& ys, const vector<double>& ts, const VectorXd& params,
                      vector<int>& warped_xs, vector<int>& warped_ys) = 0;

protected:
    string name;
    int dims;
};

class RotvelWarp : public WarpFunction {
public:
    RotvelWarp(const vector<int>& centers = {}) : WarpFunction("rotvel_warp", 1), centers(centers) {}

    void warp(const vector<int>& xs, const vector<int>& ys, const vector<double>& ts, const VectorXd& params,
              vector<int>& warped_xs, vector<int>& warped_ys) override {
        double theta = ts.back() * params[0];
        double cos_theta = cos(theta);
        double sin_theta = sin(theta);

        double xs_mean = centers.empty() ? accumulate(xs.begin(), xs.end(), 0.0) / xs.size() : centers[0];
        double ys_mean = centers.empty() ? accumulate(ys.begin(), ys.end(), 0.0) / ys.size() : centers[1];

        for (size_t i = 0; i < xs.size(); ++i) {
            double x = xs[i] - xs_mean;
            double y = ys[i] - ys_mean;
            warped_xs.push_back(static_cast<int>(x * cos_theta - y * sin_theta + xs_mean));
            warped_ys.push_back(static_cast<int>(x * sin_theta + y * cos_theta + ys_mean));
        }
    }

private:
    vector<int> centers;
};

struct OptimizationFunctor {
    OptimizationFunctor(const vector<int>& xs, const vector<int>& ys, const vector<double>& ts, WarpFunction* warp_function, ObjectiveFunction* objective,
                        const Vector2i& img_size, double blur_sigma)
        : xs(xs), ys(ys), ts(ts), warp_function(warp_function), objective(objective), img_size(img_size), blur_sigma(blur_sigma) {}

    bool operator()(const double* params, double* residual) const {
        VectorXd params_vec(1);
        params_vec << params[0];
        MatrixXd iwe;
        double loss = objective->evaluate_function(params_vec, xs, ys, ts, [this](const vector<int>& xs, const vector<int>& ys, const vector<double>& ts, const VectorXd& params, vector<int>& warped_xs, vector<int>& warped_ys) {
            warp_function->warp(xs, ys, ts, params, warped_xs, warped_ys);
        }, img_size, blur_sigma, false, iwe);
        residual[0] = loss;
        return true;
    }

    const vector<int>& xs;
    const vector<int>& ys;
    const vector<double>& ts;
    WarpFunction* warp_function;
    ObjectiveFunction* objective;
    Vector2i img_size;
    double blur_sigma;
};

VectorXd optimize(const vector<int>& xs, const vector<int>& ys, const vector<double>& ts, WarpFunction* warp_function, ObjectiveFunction* objective,
                  const Vector2i& img_size, double blur_sigma) {
    double initial_param = -M_PI / 18000.0;
    ceres::Problem problem;
    ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<OptimizationFunctor, 1, 1>(
        new OptimizationFunctor(xs, ys, ts, warp_function, objective, img_size, blur_sigma));
    problem.AddResidualBlock(cost_function, nullptr, &initial_param);

    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    cout << summary.FullReport() << endl;
    VectorXd result(1);
    result << initial_param;
    return result;
}

void read_txt_xypt(const string& file_path, vector<int>& xs, vector<int>& ys, vector<double>& ts) {
    ifstream file(file_path);
    string line;
    while (getline(file, line)) {
        stringstream ss(line);
        int x, y;
        double t;
        ss >> x >> y >> t;
        xs.push_back(x);
        ys.push_back(y);
        ts.push_back(t);
    }
}

int main() {
    Vector2i img_size(720, 1280);
    SosObjective obj;

    for (int keyname = 1; keyname <= 2; ++keyname) {
        string basedir = "./";
        string savedir = "test.csv";

        cout << "Processing " << keyname << "..." << endl;

        double blur = 0.0;
        vector<int> centers = {600, 411};

        vector<int> xs, ys;
        vector<double> ts;
        read_txt_xypt(basedir + "1_sub.txt", xs, ys, ts);

        auto start = chrono::high_resolution_clock::now();
        RotvelWarp warp(centers);
        VectorXd argmax = optimize(xs, ys, ts, &warp, &obj, img_size, blur);
        auto end = chrono::high_resolution_clock::now();

        double rpm = argmax(0) * 180.0 / M_PI / 360.0 * 60.0;
        cout << "RPM: " << abs(rpm) << endl;

        chrono::duration<double> delta_t = end - start;
        cout << "Latency: " << delta_t.count() << "s" << endl;
    }

    return 0;
}