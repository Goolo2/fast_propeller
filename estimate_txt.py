import argparse
import numpy as np
import scipy.optimize as opt
from scipy.ndimage.filters import gaussian_filter
from scipy.spatial import distance
import os
from sklearn.cluster import KMeans, DBSCAN
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
            warpfunc=None, img_size=None, blur_sigma=None, showimg=False, iwe=None):
        """
        Loss given by g(x)^2 where g(x) is IWE
        """
        if iwe is None:
            iwe = get_iwe(params, xs, ys, ts, warpfunc, img_size, use_polarity=self.use_polarity, compute_gradient=False)
        blur_sigma=self.default_blur if blur_sigma is None else blur_sigma
        # return np.random.rand()

        sos = np.mean(iwe*iwe)#/num_pix
        return -sos

def events_bounds_mask(xs, ys, x_min, x_max, y_min, y_max):
    """
    Get a mask of the events that are within the given bounds
    """
    mask = np.where(np.logical_or(xs<=x_min, xs>x_max), 0.0, 1.0)
    mask *= np.where(np.logical_or(ys<=y_min, ys>y_max), 0.0, 1.0)
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
    xt, yt, pt = torch.from_numpy(xn), torch.from_numpy(yn), torch.from_numpy(pn)
    xs, ys, ps, = xt.float(), yt.float(), pt.float()
    if device is None:
        device = xs.device
    if padding:
        img_size = (sensor_size[0]+1, sensor_size[1]+1)
    else:
        img_size = sensor_size

    mask = torch.ones(xs.size())
    if clip_out_of_range:
        zero_v = torch.tensor([0.])
        ones_v = torch.tensor([1.])
        clipx = img_size[1] if interpolation is None and padding==False else img_size[1]-1
        clipy = img_size[0] if interpolation is None and padding==False else img_size[0]-1
        mask = torch.where(xs>=clipx, zero_v, ones_v)*torch.where(ys>=clipy, zero_v, ones_v)

    pxs = xs.floor()
    pys = ys.floor()
    dxs = xs-pxs
    dys = ys-pys
    pxs = (pxs*mask).long()
    pys = (pys*mask).long()
    masked_ps = ps*mask
    img = torch.zeros(img_size).to(device)
    interpolate_to_image(pxs, pys, dxs, dys, masked_ps, img)


    return img.numpy()


def get_iwe(params, xs, ys, ts, warpfunc, img_size, compute_gradient=False, use_polarity=True):
    """
    Given a set of parameters, events and warp function, get the warped image and derivative image
    if required.
    """
    xs, ys, jx, jy = warpfunc.warp(xs, ys, ts, ts[-1], params)
    mask = events_bounds_mask(xs, ys, 0, img_size[1], 0, img_size[0])
    xs, ys = xs*mask, ys*mask
    
    ps = np.ones_like(xs)

    iwe = events_to_image_drv(xs, ys, ps, jx, jy, sensor_size=img_size,
            interpolation='bilinear', compute_gradient=compute_gradient)
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
    def __init__(self,centers=None):
        warp_function.__init__(self, 'rotvel_warp', 1)
        self.centers = centers

    def warp(self, xs, ys, ts, t0, params):
        dt = ts-t0
        theta = dt*params[0]

        xs_copy = xs.copy()
        ys_copy = ys.copy()
        # ts_copy = ts.copy()
        
        if self.centers:
            xs_mean = self.centers[0]
            ys_mean = self.centers[1]
        else:
            xs_mean = np.mean(xs_copy)
            ys_mean = np.mean(ys_copy)
        
        xs_copy = xs_copy - xs_mean
        ys_copy = ys_copy - ys_mean
        
        # rotation matrix
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        xs_tmp = xs_copy*cos_theta - ys_copy*sin_theta
        ys_tmp = xs_copy*sin_theta + ys_copy*cos_theta
        
        x_prime = xs_tmp + xs_mean
        y_prime = ys_tmp + ys_mean

        jacobian_x, jacobian_y = None, None

        return x_prime.copy(), y_prime.copy(), jacobian_x, jacobian_y
    
    
    
def optimize(xs, ys, ts, warp_function, objective, optimizer=opt.fmin_bfgs, x0=None, blur_sigma=None, img_size=(180, 240)):
    args = (xs, ys, ts, warp_function, img_size, blur_sigma)
    x0 = np.array([-np.deg2rad(-18000)])
    # bounds = (np.array([-np.deg2rad(20000)]), np.array([-np.deg2rad(-20000)]))
    bounds = (np.array([-np.deg2rad(20000)]), np.array([-np.deg2rad(7000)]))
    print(bounds, x0)
    if x0 is None:
        x0 = np.zeros(warp_function.dims)
    argmax = opt.minimize_scalar(objective.evaluate_function, args=args, bounds=bounds, method='bounded')

    print(argmax.x)
    return argmax.x



def get_data(data):
    xs = data[:, 0].astype(int)
    ys = data[:, 1].astype(int)
    ts = data[:, 2].astype(int)
    
    ts = ts-ts[0]
    ts = ts / 1e6

    print(len(xs))
    return xs, ys, ts



def calculate_rpm(data, img_size, obj, blur, centers):
    xs, ys, ts = get_data(data)

    warp = rotvel_warp(centers) #!############### very important #############################
    # 
    argmax = optimize(xs, ys, ts, warp, obj, img_size=img_size, blur_sigma=blur)
    
    rpm = np.rad2deg(argmax)/360*60

    return rpm.tolist()[0], argmax


def read_txt_xypt(file_path, max_rows=None):
    data = np.loadtxt(file_path, max_rows=max_rows)
    print(data.shape)
    x_raw = data[:, 0]
    y_raw = data[:, 1]
    p_raw = data[:, 2]
    time_raw = data[:, 3]
    
    new_data = np.column_stack((x_raw, y_raw, time_raw))
    print(new_data.shape)
    return new_data

if __name__ == "__main__":
    img_size = (720, 1280)

    obj = sos_objective()

    '''calculate rpm for multiple blades'''

    for keyname in range(1,2,1):        
        # basedir = f'H:\propeller_det_data\\testbed_exp\data\event\\processed\diff_illumination/{keyname}'
        basedir = f'./'
        # savedir = f'./results/ours/diff_illumination/{keyname}_filterd_{obj.name}_latency.csv'
        savedir = f'test.csv'
        
        print(f'Processing {keyname}...')

        # blur = 2.0 
        blur = 0.0 
        centers = [600,411]
        
        rpms_res = []
        filepath = os.path.join(basedir, '1_sub.txt')
        alldata = read_txt_xypt(filepath)
        idx = 1
        data = alldata[:2000]
        begin = time.time()
        rpm, rad_s = calculate_rpm(data, img_size, obj, blur, centers)
        end = time.time()
        
        # print(idx/len(alldata)*100.0, rad_s)
        print(f' {idx}/{len(alldata)}:  rpm = {abs(rpm)}')

        timestamp = idx * 50e3
        delta_t = end - begin
        print(f'Latency: {delta_t:.6f}s')
        rpms_res.append([timestamp, abs(rpm), delta_t])

        
        # df_rpms = pd.DataFrame(rpms_res, columns=['Time', 'RPM', 'DeltaT'])
        # df_rpms.to_csv(savedir, index=False)
        

