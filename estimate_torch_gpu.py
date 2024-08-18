import argparse
import numpy as np
import scipy.optimize as opt
from scipy.ndimage.filters import gaussian_filter
from scipy.spatial import distance
import os
# from sklearn.cluster import KMeans, DBSCAN
import pandas as pd
import time
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import torch
# import nlopt


class objective_function(ABC):

    def __init__(self, name="template", use_polarity=True,
            has_derivative=True, default_blur=1.0):
        self.name = name
        self.use_polarity = use_polarity
        self.has_derivative = has_derivative
        self.default_blur = default_blur
        super().__init__()
        
class sos_objective(objective_function):
    """
    Sum of squares objective (Stoffregen et al, Event Cameras, Contrast
    Maximization and Reward Functions: an Analysis, CVPR19)
    """

    def __init__(self):
        self.use_polarity = True
        self.name = "sos"
        self.has_derivative = True
        self.default_blur=1.0

    def evaluate_function(self, params=None, xs=None, ys=None, ts=None,
            warpfunc=None, img_size=None, blur_sigma=None, showimg=False, iwe=None, device='cpu'):
        """
        Loss given by g(x)^2 where g(x) is IWE
        """
        if iwe is None:
            iwe = get_iwe(params, xs, ys, ts, warpfunc, img_size, use_polarity=self.use_polarity, compute_gradient=False, device=device)
        blur_sigma=self.default_blur if blur_sigma is None else blur_sigma

        sos = torch.mean(iwe*iwe)
        return -sos


def events_bounds_mask(xs, ys, x_min, x_max, y_min, y_max, device='cpu'):
    """
    Get a mask of the events that are within the given bounds
    """
    mask = torch.where(torch.logical_or(xs<=x_min, xs>x_max), torch.tensor(0.0).to(device), torch.tensor(1.0).to(device))
    mask *= torch.where(torch.logical_or(ys<=y_min, ys>y_max), torch.tensor(0.0).to(device), torch.tensor(1.0).to(device))
    return mask


def interpolate_to_image(pxs, pys, dxs, dys, weights, img):
    """
    Accumulate x and y coords to an image using bilinear interpolation
    """
    img.index_put_((pys,   pxs  ), weights*(1.0-dxs)*(1.0-dys), accumulate=True)
    img.index_put_((pys,   pxs+1), weights*dxs*(1.0-dys), accumulate=True)
    img.index_put_((pys+1, pxs  ), weights*(1.0-dxs)*dys, accumulate=True)
    img.index_put_((pys+1, pxs+1), weights*dxs*dys, accumulate=True)
    return img


def interpolate_to_derivative_img(pxs, pys, dxs, dys, d_img, w1, w2):
    """
    Accumulate x and y coords to an image using double weighted bilinear interpolation
    """
    for i in range(d_img.shape[0]):
        d_img[i].index_put_((pys,   pxs  ), w1[i] * (-(1.0-dys)) + w2[i] * (-(1.0-dxs)), accumulate=True)
        d_img[i].index_put_((pys,   pxs+1), w1[i] * (1.0-dys)    + w2[i] * (-dxs), accumulate=True)
        d_img[i].index_put_((pys+1, pxs  ), w1[i] * (-dys)       + w2[i] * (1.0-dxs), accumulate=True)
        d_img[i].index_put_((pys+1, pxs+1), w1[i] * dys          + w2[i] *  dxs, accumulate=True)
        
        
def events_to_image_drv(xn, yn, pn, jacobian_xn, jacobian_yn,
        device=None, sensor_size=(180, 240), clip_out_of_range=True,
        interpolation=None, padding=True, compute_gradient=False):
    xt, yt, pt = xn, yn, pn
    xs, ys, ps, = xt.float(), yt.float(), pt.float()

    if device is None:
        device = xs.device
    if padding:
        img_size = (sensor_size[0]+1, sensor_size[1]+1)
    else:
        img_size = sensor_size

    mask = torch.ones(xs.size()).to(device)
    if clip_out_of_range:
        zero_v = torch.tensor([0.]).to(device)
        ones_v = torch.tensor([1.]).to(device)
        clipx = img_size[1] if interpolation is None and padding==False else img_size[1]-1
        clipy = img_size[0] if interpolation is None and padding==False else img_size[0]-1
        mask = torch.where(xs>=clipx, zero_v, ones_v)*torch.where(ys>=clipy, zero_v, ones_v)

    pxs = xs.floor()
    pys = ys.floor()
    dxs = xs-pxs
    dys = ys-pxs
    pxs = (pxs*mask).long()
    pys = (pys*mask).long()
    masked_ps = ps*mask
    img = torch.zeros(img_size).to(device)
    interpolate_to_image(pxs, pys, dxs, dys, masked_ps, img)

    return img


def get_iwe(params, xs, ys, ts, warpfunc, img_size, compute_gradient=False, use_polarity=True, device='cpu'):
    xs, ys, jx, jy = warpfunc.warp(xs, ys, ts, ts[-1], params, device=device)
    mask = events_bounds_mask(xs, ys, 0, img_size[1], 0, img_size[0], device=device)
    xs, ys = xs*mask, ys*mask
    
    ps = torch.ones_like(xs).to(device)
    iwe = events_to_image_drv(xs, ys, ps, jx, jy, sensor_size=img_size,
            interpolation='bilinear', compute_gradient=compute_gradient, device=device)
    return iwe


class warp_function(ABC):

    def __init__(self, name, dims):
        self.name = name
        self.dims = dims
        super().__init__()
        

class rotvel_warp(warp_function):
    """
    This class implements rotational velocity warping
    """
    def __init__(self, centers=None):
        warp_function.__init__(self, 'rotvel_warp', 1)
        self.centers = centers

    def warp(self, xs, ys, ts, t0, params, device='cpu'):
        xs_copy = torch.from_numpy(xs).to(device)
        ys_copy = torch.from_numpy(ys).to(device)
        ts_copy = torch.from_numpy(ts).to(device)
        t0 = torch.tensor(t0).to(device)
        dt = ts_copy-t0
        theta = dt*params[0]

        if self.centers:
            xs_mean = self.centers[0]
            ys_mean = self.centers[1]
        else:
            xs_mean = torch.mean(xs_copy)
            ys_mean = torch.mean(ys_copy)
        
        xs_copy = xs_copy - xs_mean
        ys_copy = ys_copy - ys_mean
        
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        xs_tmp = xs_copy*cos_theta - ys_copy*sin_theta
        ys_tmp = xs_copy*sin_theta + ys_copy*cos_theta
        
        x_prime = xs_tmp + xs_mean
        y_prime = ys_tmp + ys_mean

        jacobian_x, jacobian_y = None, None

        return x_prime, y_prime, jacobian_x, jacobian_y


def optimize(xs, ys, ts, warp_function, objective, x0=None, blur_sigma=None, img_size=(180, 240), lr=0.01, num_epochs=100, device='cpu'):
    x0 = torch.tensor([-np.deg2rad(12000)], requires_grad=True, device=device)

    optimizer = torch.optim.SGD([x0], lr=lr)
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        loss_value = objective.evaluate_function(params=x0, xs=xs, ys=ys, ts=ts, warpfunc=warp_function, img_size=img_size, blur_sigma=blur_sigma, device=device)
        loss = loss_value

        loss.backward()
        
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss_value.item()}')
    
    return x0.detach().cpu().numpy()


def get_data(data):
    xs = data[:, 0].astype(int)
    ys = data[:, 1].astype(int)
    ts = data[:, 2].astype(int)
    
    ts = ts-ts[0]
    ts = ts / 1e6

    print(len(xs))
    return xs, ys, ts


def calculate_rpm(data, img_size, obj, blur, centers, device='cpu'):
    xs, ys, ts = get_data(data)

    warp = rotvel_warp(centers)
    argmax = optimize(xs, ys, ts, warp, obj, img_size=img_size, blur_sigma=blur, device=device)
    
    rpm = np.rad2deg(argmax)/360*60

    return rpm.tolist()[0], argmax


def custom_optimize(data, img_size, obj, device='cpu'):
    xs, ys, ts = get_data(data)
    warp = rotvel_warp([600, 411])
    losses = []
    for i in range(-50, 50):
        params = np.array([-np.deg2rad(i*1000)])
        objval = obj.evaluate_function(params, xs, ys, ts, warp, img_size, device=device)
        losses.append((i, params[0], objval))
        print(i, params[0], objval)
    
    min_loss_i, rad, min_loss_value = min(losses, key=lambda x: x[2])
    rpm = min_loss_i*1000/360*60
    print('------------best------------')
    print(min_loss_i, rad, min_loss_value, rpm)
    
    losses = np.array(losses)
    plt.scatter(losses[:, 0], losses[:, 2], color='g', s=5)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", nargs='+', type=float, default=(0, 0))
    parser.add_argument("--img_size", nargs='+', type=float, default=(720, 1280))  # x,y
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])  # Device choice
    args = parser.parse_args()

    img_size = tuple(args.img_size)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(device)
    obj = sos_objective()

    for keyname in range(1, 2, 1):
        basedir = f'./'
        savedir = f'test.csv'
        
        print(f'Processing {keyname}...')

        blur = 0.0
        centers = [600, 411]
        
        rpms_res = []
        filepath = os.path.join(basedir, f'cluster_1.npy')
        alldata = np.load(filepath, allow_pickle=True)
        for idx, data in enumerate(alldata):
            begin = time.time()
            rpm, rad_s = calculate_rpm(data, img_size, obj, blur, centers, device=device)
            end = time.time()
            
            print(f' {idx}/{len(alldata)}:  rpm = {abs(rpm)}')

            timestamp = idx * 50e3
            delta_t = end - begin
            print(f'Latency: {delta_t:.6f}s')
            rpms_res.append([timestamp, abs(rpm), delta_t])

        df_rpms = pd.DataFrame(rpms_res, columns=['Time', 'RPM', 'DeltaT'])
        df_rpms.to_csv(savedir, index=False)