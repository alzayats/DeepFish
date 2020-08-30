import numpy as np
# from scipy.ndimage.interpolation import zoom

import pylab as plt
import torch 
import PIL
import scipy
import cv2
# from mlkit import utils as ut
from src import mlkit

def imread(fname):
    return scipy.misc.imread(fname)

def vis_figure(fig, win="tmp"):
    import visdom
    
    fig.savefig("tmp.jpg")
    img = l2f(imread("tmp.jpg"))
    print(img.shape)
    vis = visdom.Visdom(port=1111)
    options = dict(title=win)
    images(img, win=win, env='main', title=win) 
    plt.close()

def images(imgs, mask=None, heatmap=None, label=False, enlarge=0, 
    win="9999", nrow=4, gray=False, env="main", denorm=0,
    title=None):

    import visdom
    vis = visdom.Visdom(port=1111)

    """
    Display images into the Visdom server
    """
    # Break dict into key -> image list
    if isinstance(imgs, dict):
        for k, img in zip(imgs.keys(), imgs.values()):
            image(img, mask, label, enlarge, str(k), nrow, env,
                  vis=vis, title=title)

    # Break list into set of images
    elif isinstance(imgs, list):
        for k, img in enumerate(imgs):
            image(img, mask, label, enlarge, "%s-%d"%(win,k), 
                  nrow, env, vis=vis, title=title)

    elif isinstance(imgs, plt.Figure):
        image(f2n(imgs), mask, label, enlarge, win, nrow, env, 
             gray=gray, vis=vis, denorm=denorm, title=title)

    else:
        if heatmap is not None:
            imgs = t2n(imgs)*0.4 + 0.6*t2n(gray2cmap(heatmap))

        image(imgs, mask, label, enlarge, win, nrow, env, 
             gray=gray, vis=vis, denorm=denorm, title=title)


def image(imgs, mask, label, enlarge, win, nrow, env="main",
          vis=None, gray=False, denorm=0, title=None):
    
    if title is None:
        title = win
    imgs = get_image(imgs, mask, label, enlarge, gray,denorm)
    options = dict(title=title, xtick=True, ytick=True)

    
    vis.images(imgs, opts=options, nrow=nrow, win=win, 
               env=env)


def get_image(imgs, mask=None, label=False, enlarge=0, gray=False,
    denorm=0):
    if isinstance(imgs, PIL.Image.Image):
        imgs = np.array(imgs)
    if isinstance(mask, PIL.Image.Image):
        mask = np.array(mask)

    imgs = t2n(imgs).copy()
    imgs = l2f(imgs)

    
    # Make sure it is 4-dimensional
    if imgs.ndim == 3:
        imgs = imgs[np.newaxis]

    return imgs



def t2n(x):
    if isinstance(x, (int, float)):
        return x
    if isinstance(x, torch.autograd.Variable):
        x = x.cpu().data.numpy()
        
    if isinstance(x, (torch.cuda.FloatTensor, torch.cuda.IntTensor, torch.cuda.LongTensor, torch.cuda.DoubleTensor )):
        x = x.cpu().numpy()

    if isinstance(x, (torch.FloatTensor, torch.IntTensor, torch.LongTensor, torch.DoubleTensor )):
        x = x.numpy()

    return x

def f2n( fig ):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()
      
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))


def gray2cmap(gray, cmap="jet", thresh=0):
    # Gray has values between 0 and 255 or 0 and 1
    gray = t2n(gray)
    gray = gray / gray.max()
    gray = np.maximum(gray - thresh, 0)
    gray = gray / gray.max()
    gray = gray * 255

    gray = gray.astype(int)
    #print(gray)
   
    from pylab import get_cmap
    cmap = get_cmap(cmap)

    output = np.zeros(gray.shape + (3,), dtype=np.float64)

    for c in np.unique(gray):
        output[(gray==c).nonzero()] = cmap(c)[:3]

    return l2f(output)



def l2f(X):
    if X.ndim == 3 and (X.shape[0] == 3 or X.shape[0] == 1):
        return X
    if X.ndim == 4 and (X.shape[1] == 3 or X.shape[1] == 1):
        return X

    if X.ndim == 4 and (X.shape[1] < X.shape[3]):
        return X

    # CHANNELS LAST
    if X.ndim == 3:
        return np.transpose(X, (2,0,1))
    if X.ndim == 4:
        return np.transpose(X, (0,3,1,2))

    return X

def visualize_GT(batch, savedir, type="train"):
    pass
#     if type == "train":
#         index = "train" + str(batch["meta"]["index"].item())
#     if type == "val":
#         index = "val" + str(batch["meta"]["index"].item())

#     path_base = "%s/%s/" % (savedir, "images")
#     ut.create_dirs(path_base)
#     image = np.rollaxis(np.asarray(batch['images'][0]), 0,3)*255
#     points = np.asarray(batch['points'][0])
#     overlay = image.copy()
#     img_overlay =overlay
#     y, x = np.where(points == 1)
#     for x_cent, y_cent in zip(x,y):
#         img_overlay = cv2.circle(overlay, (x_cent,y_cent), 15, (0,0,255), -1) #
#     alpha = 0.7
#     img_gt = cv2.addWeighted(img_overlay, alpha, image, 1 - alpha, 0)
#     save_img(img_gt, path_base, index)


def save_img(img, path_base, name):
    fname_jpg = path_base + "_%s.jpg" % name
    mlkit.imsave(fname_jpg, img)
