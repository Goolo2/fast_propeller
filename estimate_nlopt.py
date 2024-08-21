import cv2
import numpy as np
import nlopt
import time

# 辅助函数，将弧度转换为度
def rad2deg(radians):
    return radians * (180.0 / np.pi)

def events_bounds_mask(xs, ys, x_min, x_max, y_min, y_max):
    mask = np.ones(xs.size)
    mask[(xs <= x_min) | (xs > x_max) | (ys <= y_min) | (ys > y_max)] = 0.0
    return mask

def events_to_image_drv(xs, ys, ps, img_size=(720, 1280), clip_out_of_range=True, padding=True):
    pxs = np.floor(xs).astype(int)
    pys = np.floor(ys).astype(int)
    img = np.zeros(img_size)

    for i in range(pxs.size):
        img[pys[i], pxs[i]] += 1

    # img_cv = img.astype(np.uint8)
    # img_cv[img_cv > 0] = 255
    # cv2.circle(img_cv, (600, 411), 5, (255, 0, 255), -1)
    # cv2.imshow("Event Image", img_cv)
    # cv2.waitKey(10)

    return img

def warp2iwe(theta, xs, ys, centers=(411,600)):
    xs_mean = centers[0]
    ys_mean = centers[1]

    xs_copy = xs - xs_mean
    ys_copy = ys - ys_mean

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    xs_tmp = xs_copy * cos_theta - ys_copy * sin_theta
    ys_tmp = xs_copy * sin_theta + ys_copy * cos_theta

    x_prime = xs_tmp + xs_mean
    y_prime = ys_tmp + ys_mean

    return x_prime, y_prime

# 定义优化目标函数
def objective_function(x, grad, data):
    xs, ys, ts, centers = data

    num_points = xs.size
    theta = (ts - ts[-1]) * x[0]

    x_prime, y_prime = warp2iwe(theta, xs, ys, centers)

    mask = events_bounds_mask(x_prime, y_prime, 0, 1280, 0, 720)
    x_prime *= mask
    y_prime *= mask

    ps = np.ones(x_prime.size)
    iwe = events_to_image_drv(x_prime, y_prime, ps, (720, 1280), True, False)

    sos = np.sum(iwe[iwe > 0]**2)
    cnt = np.sum(iwe > 0)

    # print(x[0],sos/cnt)
    return sos / cnt

def optimize(xs, ys, ts, centers):
    opt = nlopt.opt(nlopt.GN_DIRECT_L, 1)

    lb = [-600]
    ub = [-122]
    opt.set_lower_bounds(lb)
    opt.set_upper_bounds(ub)

    data = (xs, ys, ts, centers)
    opt.set_max_objective(lambda x, grad: objective_function(x, grad, data))

    opt.set_xtol_rel(1)
    opt.set_maxeval(100)

    x = [-500]
    start_time = time.time()
    minf = opt.optimize(x)
    print(minf)

    elapsed_time = time.time() - start_time
    print(f"Latency: {elapsed_time} seconds")

    rpm = rad2deg(minf) / 360 * 60
    print(f"Result: {opt.last_optimize_result()} Final parameter: {minf} rad: {rad2deg(minf)} rpm: {rpm}")

def read_txt_xyt(filename, max_rows=-1):
    xs, ys, ts = [], [], []
    with open(filename, 'r') as file:
        for count, line in enumerate(file):
            if max_rows != -1 and count >= max_rows:
                break
            x, y, p, t = map(float, line.split())
            xs.append(x)
            ys.append(y)
            ts.append(t / 1e6)

    return np.array(xs), np.array(ys), np.array(ts)



if __name__ == "__main__":
    # xs, ys, ts = read_txt_xyt("/home/lxy/CLionProjects/RotationalSpeed/1_sub.txt", 2000)
    xs, ys, ts = read_txt_xyt("./1_sub.txt", 2000)
    centers = (600, 411)
    optimize(xs, ys, ts, centers)
