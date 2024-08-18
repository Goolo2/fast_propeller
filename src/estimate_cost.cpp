#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <ceres/ceres.h>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <ctime>
#include <numeric>  // for accumulate

using namespace std;
using namespace ceres;

// Utility function to convert radians to degrees
inline double rad2deg(double radians) {
    return radians * (180.0 / M_PI);
}

// Define the objective function base class
class ObjectiveFunction {
public:
    string name;
    bool use_polarity;
    bool has_derivative;
    float default_blur;
    int dims;

    ObjectiveFunction(string name="template", bool use_polarity=true, bool has_derivative=true, float default_blur=1.0, int dims=1)
        : name(name), use_polarity(use_polarity), has_derivative(has_derivative), default_blur(default_blur), dims(dims) {}

    virtual double EvaluateFunction(const vector<double>& params, const vector<double>& xs, const vector<double>& ys, const vector<double>& ts,
                                    const vector<double>& warpfunc, const cv::Size& img_size, double blur_sigma = 1.0, bool showimg = false, const cv::Mat& iwe = cv::Mat()) = 0;
};

// Define the Sum of Squares objective function
class SosObjective : public ObjectiveFunction {
public:
    SosObjective() : ObjectiveFunction("sos", true, true, 1.0, 1) {}

    double EvaluateFunction(const vector<double>& params, const vector<double>& xs, const vector<double>& ys, const vector<double>& ts,
                            const vector<double>& warpfunc, const cv::Size& img_size, double blur_sigma=1.0, bool showimg=false, const cv::Mat& iwe=cv::Mat()) override;
};

vector<double> EventsBoundsMask(const vector<double>& xs, const vector<double>& ys, double x_min, double x_max, double y_min, double y_max) {
    vector<double> mask(xs.size(), 1.0);
    for (size_t i = 0; i < xs.size(); ++i) {
        if (xs[i] <= x_min || xs[i] > x_max || ys[i] <= y_min || ys[i] > y_max) {
            mask[i] = 0.0;
        }
    }
    return mask;
}

cv::Mat EventsToImageDrv(const vector<double>& xs, const vector<double>& ys, const vector<double>& ps, const cv::Size& img_size) {
    cv::Mat img = cv::Mat::zeros(img_size, CV_32F);

    for (size_t i = 0; i < xs.size(); ++i) {
        int px = static_cast<int>(xs[i]);
        int py = static_cast<int>(ys[i]);
        img.at<float>(py, px) += ps[i];
    }

    return img;
}

cv::Mat GetIwe(const vector<double>& params, const vector<double>& xs, const vector<double>& ys, const vector<double>& ts,
               const vector<double>& warpfunc, const cv::Size& img_size, bool compute_gradient=false, bool use_polarity=true) {
    vector<double> mask = EventsBoundsMask(xs, ys, 0, img_size.width, 0, img_size.height);

    vector<double> pxs(xs.size()), pys(ys.size()), ps(xs.size(), 1.0);

    for (size_t i = 0; i < xs.size(); ++i) {
        pxs[i] = xs[i] * mask[i];
        pys[i] = ys[i] * mask[i];
    }

    return EventsToImageDrv(pxs, pys, ps, img_size);
}

double SosObjective::EvaluateFunction(const vector<double>& params, const vector<double>& xs, const vector<double>& ys, const vector<double>& ts,
                                      const vector<double>& warpfunc, const cv::Size& img_size, double blur_sigma, bool showimg, const cv::Mat& iwe) {
    blur_sigma = default_blur;

    // Compute the IWE
    cv::Mat iwe_computed = GetIwe(params, xs, ys, ts, warpfunc, img_size, false, use_polarity);
    double sos = cv::mean(iwe_computed.mul(iwe_computed))[0];

    return -sos;
}

// Warp function base class
class WarpFunction {
public:
    string name;
    int dims;

    WarpFunction(string name, int dims) : name(name), dims(dims) {}
};

// Rotational velocity warp
class RotVelWarp : public WarpFunction {
public:
    vector<double> centers;

    RotVelWarp(const vector<double>& centers = {}): WarpFunction("rotvel_warp", 1), centers(centers) {}

    void Warp(vector<double>& xs, vector<double>& ys, const vector<double>& ts, double t0, vector<double> params) {
        double theta = (ts.back() - t0) * params[0];

        double cos_theta = cos(theta);
        double sin_theta = sin(theta);

        double xs_mean = centers.empty() ? accumulate(xs.begin(), xs.end(), 0.0)/xs.size() : centers[0];
        double ys_mean = centers.empty() ? accumulate(ys.begin(), ys.end(), 0.0)/ys.size() : centers[1];

        for (size_t i = 0; i < xs.size(); ++i) {
            xs[i] -= xs_mean;
            ys[i] -= ys_mean;
            double xs_tmp = xs[i] * cos_theta - ys[i] * sin_theta;
            double ys_tmp = xs[i] * sin_theta + ys[i] * cos_theta;
            xs[i] = xs_tmp + xs_mean;
            ys[i] = ys_tmp + ys_mean;
        }
    }
};

// Objective function cost function for Ceres
// class ObjectiveFunctionCost : public ceres::FirstOrderFunction {
// public:
//     ObjectiveFunction *objective_function;
//     vector<double> xs, ys, ts, warpfunc;
//     cv::Size img_size;

//     ObjectiveFunctionCost(ObjectiveFunction* obj, const vector<double>& xs, const vector<double>& ys, const vector<double>& ts, const vector<double>& warpfunc, const cv::Size& img_size)
//         : objective_function(obj), xs(xs), ys(ys), ts(ts), warpfunc(warpfunc), img_size(img_size) {}

//     virtual ~ObjectiveFunctionCost() {}

//     bool Evaluate(const double* parameters, double* cost, double* gradient) const override {
//         vector<double> params(parameters, parameters + objective_function->dims);
//         *cost = objective_function->EvaluateFunction(params, xs, ys, ts, warpfunc, img_size, 1.0, false, cv::Mat());
//         return true;
//     }

//     int NumParameters() const override { return objective_function->dims; }
// };
// 定义目标函数的残差类
class ObjectiveFunctionCost : public ceres::CostFunction {
public:
    ObjectiveFunctionCost(ObjectiveFunction* obj,
                          const vector<double>& xs,
                          const vector<double>& ys,
                          const vector<double>& ts,
                          const vector<double>& warpfunc,
                          const cv::Size& img_size)
        : objective_function(obj), xs(xs), ys(ys), ts(ts), warpfunc(warpfunc), img_size(img_size) {
        // 设置 CostFunction 的参数块大小
        set_num_residuals(1);
        mutable_parameter_block_sizes()->push_back(objective_function->dims);
    }

    virtual ~ObjectiveFunctionCost() {}

    virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override {
        vector<double> params(parameters[0], parameters[0] + objective_function->dims);
        residuals[0] = objective_function->EvaluateFunction(params, xs, ys, ts, warpfunc, img_size, 1.0, false, cv::Mat());

        // 这里忽略了 Jacobian 的计算，你可以根据需要实现它。
        if (jacobians != nullptr && jacobians[0] != nullptr) {
            for (int i = 0; i < objective_function->dims; ++i) {
                jacobians[0][i] = 0.0; // Placeholders, should be replaced with actual values.
            }
        }

        return true;
    }

private:
    ObjectiveFunction *objective_function;
    vector<double> xs, ys, ts, warpfunc;
    cv::Size img_size;
};

// // Optimization function
// vector<double> Optimize(vector<double> xs, vector<double> ys, vector<double> ts, RotVelWarp warp_function, ObjectiveFunction& objective, cv::Size img_size, double blur_sigma) {
//     // vector<double> initial_params(1, -M_PI / 24);
//     vector<double> initial_params(1, 315);
//     ceres::GradientProblemSolver::Options options;
//     ceres::GradientProblemSolver::Summary summary;

//     ceres::GradientProblem problem(new ObjectiveFunctionCost(&objective, xs, ys, ts, warp_function.centers, img_size));
//     ceres::Solve(options, problem, initial_params.data(), &summary);

//     return initial_params;
// }
vector<double> Optimize(vector<double> xs, vector<double> ys, vector<double> ts, RotVelWarp warp_function, ObjectiveFunction& objective, cv::Size img_size, double blur_sigma) {
    vector<double> initial_params(1, 315.0);

    // 创建一个 Ceres 问题实例
    ceres::Problem problem;

    // 创建残差块
    ObjectiveFunctionCost* cost_function = new ObjectiveFunctionCost(&objective, xs, ys, ts, warp_function.centers, img_size);

    // 添加残差块到问题实例中
    problem.AddResidualBlock(cost_function, nullptr, initial_params.data());

    // 配置求解器选项
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;

    // 存放结果摘要信息
    ceres::Solver::Summary summary;

    // 求解问题
    ceres::Solve(options, &problem, &summary);

    return initial_params;
}


// Read data from file
vector<vector<double>> ReadTxtXypt(string file_path, int max_rows=-1) {
    ifstream file(file_path);
    vector<vector<double>> alldata;
    double x, y, p, t;
    int count = 0;

    while (file >> x >> y >> p >> t && (max_rows == -1 || count < max_rows)) {
        alldata.push_back({x, y, t});
        count++;
    }

    return alldata;
}

// Main calculation function
double CalculateRpm(const vector<vector<double>>& data, cv::Size img_size, ObjectiveFunction& obj, double blur, vector<double> centers) {
    vector<double> xs(data.size()), ys(data.size()), ts(data.size());

    for(size_t i = 0; i < data.size(); i++) {
        xs[i] = data[i][0];
        ys[i] = data[i][1];
        ts[i] = data[i][2];
    }

    RotVelWarp warp(centers);
    vector<double> optimized_params = Optimize(xs, ys, ts, warp, obj, img_size, blur);

    double rpm = rad2deg(optimized_params[0]) / 360 * 60;
    cout << "optimized_params " << optimized_params[0] << endl;
    return rpm;
}

int main() {
    cv::Size img_size(720, 1280);
    SosObjective obj;
    double blur = 0.0;
    vector<double> centers = {600, 411};
    string filepath = "/home/goolo/projects/fast_propeller/1_sub.txt";
    vector<vector<double>> alldata = ReadTxtXypt(filepath, 100);

    clock_t begin = clock();
    double rpm = CalculateRpm(alldata, img_size, obj, blur, centers);
    clock_t end = clock();

    double delta_t = double(end - begin) / CLOCKS_PER_SEC;
    cout << "Processing: RPM = " << abs(rpm) << endl;
    cout << "Latency: " << delta_t << " seconds" << endl;

}