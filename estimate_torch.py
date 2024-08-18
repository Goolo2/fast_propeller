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
import nlopt


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

        # sos = np.mean(iwe*iwe)#/num_pix
        sos = torch.mean(iwe*iwe)#/num_pix
        return -sos
        
        # loss = np.var(iwe-np.mean(iwe))
        # return -loss
        
        # exp = np.exp(iwe.astype(np.double))
        # soe = np.mean(exp)
        # return -soe

def events_bounds_mask(xs, ys, x_min, x_max, y_min, y_max):
    """
    Get a mask of the events that are within the given bounds
    """
    # mask = np.where(np.logical_or(xs<=x_min, xs>x_max), 0.0, 1.0)
    # mask *= np.where(np.logical_or(ys<=y_min, ys>y_max), 0.0, 1.0)
    mask = torch.where(torch.logical_or(xs<=x_min, xs>x_max), 0.0, 1.0)
    mask *= torch.where(torch.logical_or(ys<=y_min, ys>y_max), 0.0, 1.0)
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
    # xt, yt, pt = torch.from_numpy(xn), torch.from_numpy(yn), torch.from_numpy(pn)
    xt, yt, pt = xn, yn, pn
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


    # return img.numpy()
    return img


def get_iwe(params, xs, ys, ts, warpfunc, img_size, compute_gradient=False, use_polarity=True):
    """
    Given a set of parameters, events and warp function, get the warped image and derivative image
    if required.
    """
    xs, ys, jx, jy = warpfunc.warp(xs, ys, ts, ts[-1], params)
    mask = events_bounds_mask(xs, ys, 0, img_size[1], 0, img_size[0])
    xs, ys = xs*mask, ys*mask
    
    # ps = np.ones_like(xs)
    ps = torch.ones_like(xs)

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
        xs_copy = torch.from_numpy(xs)
        ys_copy = torch.from_numpy(ys)
        ts_copy = torch.from_numpy(ts)
        t0 = torch.tensor(t0)
        dt = ts_copy-t0
        theta = dt*params[0]

        # xs_copy = xs.copy()
        # ys_copy = ys.copy()
        # ts_copy = ts.copy()
        
        if self.centers:
            xs_mean = self.centers[0]
            ys_mean = self.centers[1]
        else:
            xs_mean = torch.mean(xs_copy)
            ys_mean = torch.mean(ys_copy)
        
        xs_copy = xs_copy - xs_mean
        ys_copy = ys_copy - ys_mean
        
        # rotation matrix
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        xs_tmp = xs_copy*cos_theta - ys_copy*sin_theta
        ys_tmp = xs_copy*sin_theta + ys_copy*cos_theta
        
        x_prime = xs_tmp + xs_mean
        y_prime = ys_tmp + ys_mean

        jacobian_x, jacobian_y = None, None

        return x_prime, y_prime, jacobian_x, jacobian_y
    
    
    
# def optimize(xs, ys, ts, warp_function, objective, optimizer=opt.fmin_bfgs, x0=None, blur_sigma=None, img_size=(180, 240)):
#     args = (xs, ys, ts, warp_function, img_size, blur_sigma)
#     x0 = np.array([-np.deg2rad(12000)])
#     # bounds = (np.array([-np.deg2rad(20000)]), np.array([-np.deg2rad(-20000)]))
#     bounds = (np.array([-np.deg2rad(20000)]), np.array([-np.deg2rad(7000)]))
#     print(bounds, x0)
#     if x0 is None:
#         x0 = np.zeros(warp_function.dims)
#     argmax = opt.minimize_scalar(objective.evaluate_function, args=args, bounds=bounds, method='bounded')
#     print(argmax.x)
#     return argmax.x
    
def optimize(xs, ys, ts, warp_function, objective, x0=None, blur_sigma=None, img_size=(180, 240), lr=0.01, num_epochs=100):
    # x0 = torch.tensor(x0, requires_grad=True)
    x0 = torch.tensor([-np.deg2rad(12000)], requires_grad=True)
    # 选择优化器
    optimizer = torch.optim.SGD([x0], lr=1000)
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # 计算目标函数值
        # loss_value = objective.evaluate_function(params=x0.detach().numpy(), xs=xs, ys=ys, ts=ts, warpfunc=warp_function, img_size=img_size, blur_sigma=blur_sigma)
        loss_value = objective.evaluate_function(params=x0, xs=xs, ys=ys, ts=ts, warpfunc=warp_function, img_size=img_size, blur_sigma=blur_sigma)
        loss = torch.tensor(loss_value, requires_grad=True)
        
        # 计算梯度
        loss.backward()
        
        # 执行优化步骤
        optimizer.step()
        
        grads = x0.grad
        print(f'Gradient: {grads}')
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss_value}')
    
    return x0.detach().numpy()


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




def custom_optimize(data):
    xs, ys, ts = get_data(data)
    # ?choose objective function#########################################
    # obj = variance_objective()
    # obj = rms_objective()
    obj = sos_objective()
    # obj = soe_objective()
    # obj = moa_objective()
    # obj = isoa_objective()
    # obj = sosa_objective()
    # obj = tsmap_objective()
    # obj = timestamp_objective()
    # obj = biobj_objective()

    warp = rotvel_warp([600,411])
    losses= []
    for i in range(-50,50):
        params = np.array([-np.deg2rad(i*1000)])
        objval = obj.evaluate_function(params, xs, ys, ts, warp, img_size)
        # iwe, d_iwe = get_iwe(params, xs, ys, ts, warp, img_size, use_polarity=False, compute_gradient=False)
        # print(iwe.shape)
        # objval = -np.var(iwe-np.mean(iwe))
        
        losses.append((i, params[0], objval))
        print(i, params[0], objval)
    
    min_loss_i, rad, min_loss_value = min(losses, key=lambda x: x[2])
    rpm = min_loss_i*1000/360*60
    print('------------best------------')
    print(min_loss_i, rad, min_loss_value, rpm)
    
    losses = np.array(losses)
    plt.scatter(losses[:,0], losses[:,2], color='g', s=5)
    plt.show()
    
    
if __name__ == "__main__":
    """
    Quick demo of various objectives.
    Args:
        path Path to h5 file with event data
        gt Ground truth optic flow for event slice
        img_size The size of the event camera sensor
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", nargs='+', type=float, default=(0, 0))
    parser.add_argument("--img_size", nargs='+',
                        type=float, default=(720, 1280))  # x,y
    # parser.add_argument("--img_size", nargs='+', type=float, default=(-100,100))
    args = parser.parse_args()

    img_size = tuple(args.img_size)

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
        filepath = os.path.join(basedir, f'cluster_1.npy')
        alldata = np.load(filepath, allow_pickle=True)
        for idx, data in enumerate(alldata):
            # data = data[:1000]
            begin = time.time()
            # custom_optimize(data)
            rpm, rad_s = calculate_rpm(data, img_size, obj, blur, centers)
            end = time.time()
            
            # print(idx/len(alldata)*100.0, rad_s)
            print(f' {idx}/{len(alldata)}:  rpm = {abs(rpm)}')

            timestamp = idx * 50e3
            delta_t = end - begin
            print(f'Latency: {delta_t:.6f}s')
            rpms_res.append([timestamp, abs(rpm), delta_t])

        
        df_rpms = pd.DataFrame(rpms_res, columns=['Time', 'RPM', 'DeltaT'])
        df_rpms.to_csv(savedir, index=False)
        

