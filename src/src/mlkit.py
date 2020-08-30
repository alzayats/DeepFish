from torch.autograd import Variable
import hashlib 
import pickle
import json
import os
import itertools
import torch
import numpy as np
import PIL
import hashlib
import yaml
import pickle
import os
import torch.nn.functional as F
import torch
import time
from datetime import datetime
import pytz
import threading
import pylab as plt
import subprocess
import shlex 
import numpy as np
import contextlib
import json
import scipy
from importlib import reload
from skimage.segmentation import mark_boundaries

def get_padding(kernel_size=1):
    return int((kernel_size - 1) / 2)

# =======================================================
# exp helpers
def cartesian_exp_group(exp_config):
    """Cartesian experiment config.

    It converts the exp_config into a list of experiment configuration by doing
    the cartesian product of the different configuration. It is equivalent to
    do a grid search.

    Args:
        exp_config: Dictionary with the experiment Configuration

    Returns:
        exp_list: List with an experiment dictionary for each individual
        experiment

    """
    # Create the cartesian product
    exp_list_raw = (dict(zip(exp_config.keys(), values))
                    for values in itertools.product(*exp_config.values()))

    # Convert into a list
    exp_list = []
    for exp_dict in exp_list_raw:
        exp_list += [exp_dict]
    return exp_list

def hash_dict(dictionary):
    """Create a hash for a dictionary."""
    dict2hash = ""

    for k in sorted(dictionary.keys()):
        if isinstance(dictionary[k], dict):
            v = hash_dict(dictionary[k])
        else:
            v = dictionary[k]

        dict2hash += "%s_%s_" % (str(k), str(v))

    return hashlib.md5(dict2hash.encode()).hexdigest()

# =======================================================
# checkpoint helpers
class CheckpointManager:
    """Checkpoint manager."""

    def __init__(self, exp_dict=None, savedir=None, mins2save=1, verbose=1):
        """Constructor."""
        # Save arguments
        self.exp_dict = exp_dict
        self.savedir = savedir
        self.mins2save = mins2save
        self.verbose = verbose
        # self.wait = 120  # seconds
        # self.friend_exp_dict = friend_exp_dict

    def save(self, save_dict):
        """
        Save the checkpoint to disk.

        Args:
            save_dict: dictionary of files to save

        Returns:

        """
        # Save each field of the dictionary independently in a file
        for k, v in save_dict.items():
            # Define the filename
            fname = os.path.join(self.savedir, k)

            # Save with pytorch or pickle
            if ".pth" in k:
                torch_save(fname, v)
            else:
                save_pkl(fname, v)

    def load(self, keys):
        """Load latest checkpoint and history.

        Args:
            keys: List of filenames to load

        Returns:
            save_dict: Dictionary of the filenames with their template_content

        """
        save_dict = {}
        for k in keys:
            # Filename with the path
            fname = os.path.join(self.savedir, k)

            # Load content with pytorch or pickle
            if ".pth" in k:
                save_dict[k] = ut.torch_load(fname)
            else:
                save_dict[k] = ut.load_pkl(fname)

        return save_dict

def create_checkpoint(savedir, exp_dict):
    save_json(savedir + "/exp_dict.json", exp_dict)
    save_json(savedir + "/run_dict.json", {"started at":time_to_montreal()})
    # print("Saved: %s" % savedir)

def delete_checkpoint(savedir):
    if os.path.exists(savedir + "/run_dict.json"):
        os.remove(savedir + "/run_dict.json")
    if os.path.exists(savedir + "/meta_dict.pkl"):
        os.remove(savedir + "/meta_dict.pkl")
    if os.path.exists(savedir + "/score_list.pkl"):
        os.remove(savedir + "/score_list.pkl")


def create_tiny(dataset, size=5):
    data = [dataset[i] for i in range(size)]

    class TinyDataset(torch.nn.Module):
        def __init__(self, data):
            self.data = data

        def __getitem__(self, item):
            return self.data[item]

        def __len__(self):
            return len(self.data)

    dataset = TinyDataset(data)
    return dataset

def checkpoint_exists(savedir):
    if (os.path.exists(savedir + "/run_dict.json") and
        os.path.exists(savedir + "/score_list.pkl")):
        return True
    return False

# =======================================================
# helpers
def load_pkl(fname):
    """Load the content of a pkl file."""
    with open(fname, "rb") as f:
        return pickle.load(f)

def load_json(fname, decode=None):
    with open(fname, "r") as json_file:
        d = json.load(json_file)

    return d

# def torch_save(fname, obj):
#     """"Save data in torch format."""
#     # Define names of temporal files
#     os.makedirs(os.path.dirname(fname), exist_ok=True)
#     fname_tmp = fname + ".tmp"
#
#     torch.save(obj, fname_tmp)
#     os.rename(fname_tmp, fname)

def read_text(fname):
    # READS LINES
    with open(fname, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return lines

# def save_pkl(fname, dict):
#     os.makedirs(os.path.dirname(fname), exist_ok=True)
#     fname_tmp = fname + "_tmp.pkl"
#     with open(fname_tmp, "wb") as f:
#         pickle.dump(dict, f)
#     os.rename(fname_tmp, fname)

def save_json(fname, data):
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    with open(fname, "w") as json_file:
        json.dump(data, json_file, indent=4, sort_keys=True)

def save_pkl(fname, data, with_rename=True):
    """Save data in pkl format."""
    # Create folder
    create_dirs(fname)

    # Save file
    if with_rename:
        fname_tmp = fname + "_tmp.pth"
        with open(fname_tmp, "wb") as f:
            pickle.dump(data, f)
        if os.path.exists(fname):
            os.remove(fname)
        os.rename(fname_tmp, fname)
    else:
        with open(fname, "wb") as f:
            pickle.dump(data, f)

def torch_save(fname, obj, with_rename=True):
    """"Save data in torch format."""
    # Define names of temporal files
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    # fname_tmp = fname + ".tmp"

    # torch.save(obj, fname_tmp)
    # os.rename(fname_tmp, fname)

    if with_rename:
        fname_tmp = fname + "_tmp.pth"
        with open(fname_tmp, "wb") as f:
            torch.save(obj, f)
        if os.path.exists(fname):
            os.remove(fname)
        os.rename(fname_tmp, fname)
    else:
        with open(fname, "wb") as f:
            torch.save(obj, f)

def create_dirs(fname):
    """Create folders."""
    # If it is a filename do nothing
    if "/" not in fname:
        return

    # If the folder do not exist, create it
    if not os.path.exists(os.path.dirname(fname)):
        print(os.path.dirname(fname))
        os.makedirs(os.path.dirname(fname))


def time_to_montreal():  # TODO: Remove commented code
    """Get time in Montreal zone."""
    # Get time
    ts = time.time()

    tz = pytz.timezone('America/Montreal')
    # dt = aware_utc_dt.astimezone(tz)
    dt = datetime.fromtimestamp(ts, tz)

    return dt.strftime("%I:%M %p (%b %d)")

def n2t(x, dtype="float"):
    if isinstance(x, (int, np.int64, float)):
        x = np.array([x])

    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    return x


# ===============================================================
# image helpers
def get_image(imgs,
              mask=None,
              label=False,
              enlarge=0,
              gray=False,
              denorm=0,
              bbox_yxyx=None,
              annList=None,
              pretty=False,
              pointList=None,
              win=None,
              **options):

    if isinstance(imgs, tuple):
            fig = None

            colors = ["red", "blue"]
            for i in range(len(imgs)):
                fig = scatter_plot(imgs[i], color=colors[i], fig=fig, title=win)

            return visFigure(fig, win=win)
            

    imgs = denormalize(imgs, mode=denorm)
    if isinstance(imgs, PIL.Image.Image):
        imgs = np.array(imgs)
    if isinstance(mask, PIL.Image.Image):
        mask = np.array(mask)

    imgs = t2n(imgs).copy()
    imgs = l2f(imgs)

    if pointList is not None and len(pointList):
        h, w = pointList[0]["h"], pointList[0]["w"]
        mask_points = np.zeros((h, w))
        for p in pointList:
            y, x = p["y"], p["x"]
            mask_points[int(h*y), int(w*x)] = 1
        imgs = maskOnImage(imgs, mask_points, enlarge=1)

    if pretty or annList is not None:
        imgs = pretty_vis(imgs, annList, **options)
        imgs = l2f(imgs)

    if mask is not None and mask.sum() != 0:
        imgs = maskOnImage(imgs, mask, enlarge)

    if bbox_yxyx is not None:
        _, _, h, w = imgs.shape
        mask = bbox_yxyx_2_mask(bbox_yxyx, h, w)
        imgs = maskOnImage(imgs, mask, enlarge=1)

    # LABEL
    elif (not gray) and (label or imgs.ndim == 2 or
                         (imgs.ndim == 3 and imgs.shape[0] != 3) or
                         (imgs.ndim == 4 and imgs.shape[1] != 3)):

        imgs = label2Image(imgs)

        if enlarge:
            imgs = zoom(imgs, 11)

    # Make sure it is 4-dimensional
    if imgs.ndim == 3:
        imgs = imgs[np.newaxis]

    return imgs



def zoom(img, kernel_size=3):
    img = n2t(img)
    if img.dim() == 4:
        img = img.sum(1).unsqueeze(1)
    img = Variable(n2t(img)).float()
    img = F.max_pool2d(
        img,
        kernel_size=kernel_size,
        stride=1,
        padding=get_padding(kernel_size))
    return t2n(img)


def label2Image(imgs):
    imgs = t2n(imgs).copy()

    if imgs.ndim == 3:
        imgs = imgs[:, np.newaxis]

    imgs = l2f(imgs)

    if imgs.ndim == 4 and imgs.shape[1] != 1:
        imgs = np.argmax(imgs, 1)

    imgs = label2rgb(imgs)

    if imgs.ndim == 3:
        imgs = imgs[np.newaxis]
    return imgs

def label2Image(imgs):
    imgs = t2n(imgs).copy()

    if imgs.ndim == 3:
        imgs = imgs[:, np.newaxis]

    imgs = l2f(imgs)

    if imgs.ndim == 4 and imgs.shape[1] != 1:
        imgs = np.argmax(imgs, 1)

    imgs = label2rgb(imgs)

    if imgs.ndim == 3:
        imgs = imgs[np.newaxis]
    return imgs


def label2rgb(labels):
    from skimage.color import label2rgb
    return label2rgb(t2n(labels), bg_label=0, bg_color=(0.,0.,0.))  

def bbox_yxyx_2_mask(bbox, h, w):
    mask = np.zeros((h, w), int)
    assert bbox.max() <= 1
    bbox = t2n(bbox)
    for i in range(bbox.shape[0]):
        y1, x1, y2, x2 = bbox[i]
        # print(y1, y2, x1, x2)
        y1 = min(int(y1 * h), h - 1)
        y2 = min(int(y2 * h), h - 1)
        x1 = min(int(x1 * w), w - 1)
        x2 = min(int(x2 * w), w - 1)
        # print(y1, y2, x1, x2)
        mask[y1:y2, x1] = i + 1
        mask[y1:y2, x2] = i + 1

        mask[y1, x1:x2] = i + 1
        mask[y2, x1:x2] = i + 1

    return mask

def t2n(x):
    if isinstance(x, (int, float)):
        return x
    if isinstance(x, torch.autograd.Variable):
        x = x.cpu().data.numpy()

    if isinstance(x, (torch.cuda.FloatTensor, torch.cuda.IntTensor,
                      torch.cuda.LongTensor, torch.cuda.DoubleTensor)):
        x = x.cpu().numpy()

    if isinstance(x, (torch.FloatTensor, torch.IntTensor, torch.LongTensor,
                      torch.DoubleTensor)):
        x = x.numpy()

    return x

def scatter_plot(X, color, fig=None, title=""):
    if fig is None:
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(1, 1, 1) 

    ax = fig.axes[0]

    ax.grid(linestyle='dotted')
    ax.scatter(X[:,0],X[:,1], alpha=0.6, c=color, edgecolors="black")
    

    # plt.axes().set_aspect('equal', 'datalim')
    ax.set_title(title)
    ax.set_xlabel("t-SNE Feature 2")
    ax.set_ylabel("t-SNE Feature 1")
   
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    return fig 

def maskOnImage(imgs, mask, enlarge=0):
    imgs = l2f(t2n(imgs)).copy()
    mask = l2f(label2Image(mask))

    if enlarge:
        mask = zoom(mask, 11)

    if mask.max() > 1:
        mask = mask / 255.

    if imgs.max() > 1:
        imgs = imgs / 255.

    nz = mask.squeeze() != 0
    imgs = imgs * 0.5 + mask * 0.5
    imgs /= imgs.max()
    # print(mask.max(), imgs.shape, mask.shape)
    # ind = np.where(nz)

    # if len(ind) == 3:
    #     k, r, c = ind
    #     imgs[:,k,r,c] = imgs[:,k,r,c]*0.5 + mask[:,k,r,c] * 0.5
    #     imgs[:,k,r,c]  = imgs[:,k,r,c]/imgs[:,k,r,c].max()

    # if len(ind) == 2:
    #     r, c = ind
    #     imgs[:,:,r,c] = imgs[:,:,r,c]*0.5 + mask[:,:,r,c] * 0.5
    #     imgs[:,:,r,c]  = imgs[:,:,r,c]/imgs[:,:,r,c].max()

    #print(imgs[nz])
    #print(imgs.shape)
    #print(mask.shape)
    if mask.ndim == 4:
        mask = mask.sum(1)

    nz = mask != 0
    mask[nz] = 1

    mask = mask.astype(int)

    #imgs = imgs*0.5 + mask[:, :, :, np.newaxis] * 0.5

    segList = []
    for i in range(imgs.shape[0]):
        segList += [
            l2f(
                mark_boundaries(
                    f2l(imgs[i]).copy(), f2l(mask[i]), mode="inner"))
        ]
        # segList += [l2f(imgs[i]).copy()]
    imgs = np.stack(segList)

    return l2f(imgs)
    
def l2f(X):
    if X.ndim == 3 and (X.shape[0] == 3 or X.shape[0] == 1):
        return X
    if X.ndim == 4 and (X.shape[1] == 3 or X.shape[1] == 1):
        return X

    if X.ndim == 4 and (X.shape[1] < X.shape[3]):
        return X

    # CHANNELS LAST
    if X.ndim == 3:
        return np.transpose(X, (2, 0, 1))
    if X.ndim == 4:
        return np.transpose(X, (0, 3, 1, 2))

    return X

def _denorm(image, mu, var, bgr2rgb=False):
    if image.ndim == 3:
        result = image * var[:, None, None] + mu[:, None, None]
        if bgr2rgb:
            result = result[::-1]
    else:
        result = image * var[None, :, None, None] + mu[None, :, None, None]
        if bgr2rgb:
            result = result[:, ::-1]
    return result


def denormalize(img, mode=0):
    # _img = t2n(img)
    # _img = _img.copy()
    image = t2n(img).copy().astype("float")

    if mode in [1, "rgb"]:
        mu = np.array([0.485, 0.456, 0.406])
        var = np.array([0.229, 0.224, 0.225])
        # (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        image = _denorm(image, mu, var)

    elif mode in [2, "bgr"]:
        mu = np.array([102.9801, 115.9465, 122.7717])
        var = np.array([1, 1, 1])
        image = _denorm(image, mu, var, bgr2rgb=True).clip(0, 255).round()

    elif mode in [3, "basic"]:
        mu = np.array([0.5, 0.5, 0.5])
        var = np.array([0.5, 0.5, 0.5])

        image = _denorm(image, mu, var)
    # else:

    #     mu = np.array([0.,0.,0.])
    #     var = np.array([1,1,1])

    #     image = _denorm(image, mu, var)
    # print(image)
    return image



def pretty_vis(image, annList, show_class=False, alpha=0.0, dpi=100, **options):
    import cv2
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.patches import Polygon
    from matplotlib.figure import Figure
    from . import ann_utils as au
    # print(image)
    # if not image.as > 1:
    #     image = image.astype(float)/255.
    image = f2l(image).squeeze().clip(0, 255)
    if image.max() > 1:
        image /= 255.

    # box_alpha = 0.5
    # print(image.clip(0, 255).max())
    color_list = colormap(rgb=True) / 255.

    # fig = Figure()
    fig = plt.figure(frameon=False)
    canvas = FigureCanvas(fig)
    fig.set_size_inches(image.shape[1] / dpi, image.shape[0] / dpi)
    # ax = fig.gca()

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.axis('off')
    fig.add_axes(ax)
    # im = im.clip(0, 1)
    # print(image)
    ax.imshow(image)

    # Display in largest to smallest order to reduce occlusion
    # areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    # sorted_inds = np.argsort(-areas)

    mask_color_id = 0
    for i in range(len(annList)):
        ann = annList[i]

        # bbox = boxes[i, :4]
        # score = boxes[i, -1]

        # bbox = au.ann2bbox(ann)["shape"]
        # score = ann["score"]

        # if score < thresh:
        #     continue

        # show box (off by default, box_alpha=0.0)
        if "bbox" in ann:
            bbox = ann["bbox"]
            ax.add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2],
                              bbox[3],
                              fill=False,
                              edgecolor='r',
                              linewidth=3.0,
                              alpha=0.5))

        # if show_class:
        # if options.get("show_text") == True or options.get("show_text") is None:
        #     score = ann["score"] or -1
        #     ax.text(
        #         bbox[0], bbox[1] - 2,
        #         "%.1f" % score,
        #         fontsize=14,
        #         family='serif',
        #         bbox=dict(facecolor='g', alpha=1.0, pad=0, edgecolor='none'),
        #         color='white')

        # show mask
        if "segmentation" in ann:
            mask = au.ann2mask(ann)["mask"]
            img = np.ones(image.shape)
            # category_id = ann["category_id"]
            # mask_color_id = category_id - 1
            # color_list = ["r", "g", "b","y", "w","orange","purple"]
            # color_mask = color_list[mask_color_id % len(color_list)]
            color_mask = color_list[mask_color_id % len(color_list), 0:3]
            mask_color_id += 1
            # print("color id: %d - category_id: %d - color mask: %s" 
                        # %(mask_color_id, category_id, str(color_mask)))
            w_ratio = .4
            for c in range(3):
                color_mask[c] = color_mask[c] * (1 - w_ratio) + w_ratio
            for c in range(3):
                img[:, :, c] = color_mask[c]
            e = mask

            contour, hier = cv2.findContours(e.copy(), 
                                    cv2.RETR_CCOMP,
                                    cv2.CHAIN_APPROX_NONE)

            for c in contour:
                polygon = Polygon(
                    c.reshape((-1, 2)),
                    fill=True,
                    facecolor=color_mask,
                    edgecolor="white",
                    linewidth=1.5,
                    alpha=0.7
                    )
                ax.add_patch(polygon)

    canvas.draw()  # draw the canvas, cache the renderer
    width, height = fig.get_size_inches() * fig.get_dpi()
    # image = np.fromstring(canvas.tostring_rgb(), dtype='uint8')

    fig_image = np.fromstring(
        canvas.tostring_rgb(), dtype='uint8').reshape(
            int(height), int(width), 3)
    plt.close()
    # print(fig_image)
    return fig_image

import os
def save_image(fname, img):
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    images = f2l(t2n(img))
    imsave(fname , img)

def f2l(X):
    if X.ndim == 3 and (X.shape[2] == 3 or X.shape[2] == 1):
        return X
    if X.ndim == 4 and (X.shape[3] == 3 or X.shape[3] == 1):
        return X

    # CHANNELS FIRST
    if X.ndim == 3:
        return np.transpose(X, (1, 2, 0))
    if X.ndim == 4:
        return np.transpose(X, (0, 2, 3, 1))

    return X

def gray2cmap(gray, cmap="jet", thresh=0):
    # Gray has values between 0 and 255 or 0 and 1
    gray = t2n(gray)
    gray = gray / max(1, gray.max())
    gray = np.maximum(gray - thresh, 0)
    gray = gray / max(1, gray.max())
    gray = gray * 255

    gray = gray.astype(int)
    #print(gray)

    from pylab import get_cmap
    cmap = get_cmap(cmap)

    output = np.zeros(gray.shape + (3, ), dtype=np.float64)

    for c in np.unique(gray):
        output[(gray == c).nonzero()] = cmap(c)[:3]

    return l2f(output)

def imsave(fname, arr, size=None):
    from PIL import Image
    arr = f2l(t2n(arr)).squeeze()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    #print(arr.shape)
    if size is not None:
        arr = Image.fromarray(arr)
        arr = arr.resize(size)
        arr = np.array(arr)
    # scipy.misc.imsave(fname, arr)
    img = PIL.Image.fromarray(np.uint8(arr * 255))
    img.save(fname)


# ========================================================
# results
import zipfile

def fname_parent(filepath, levels=1):
    common = filepath
    for i in range(levels + 1):
        common = os.path.dirname(common)
    return os.path.relpath(filepath, common)
    
def zipdir(src_dirname, out_fname, include_list=None):
    zipf = zipfile.ZipFile(out_fname, 'w', 
                           zipfile.ZIP_DEFLATED)
    # ziph is zipfile handle
    for root, dirs, files in os.walk(src_dirname):
        for file in files:
            
            if include_list is not None and file not in include_list:
                continue
            
            abs_path = os.path.join(root, file)
            rel_path = fname_parent(abs_path)
            print(rel_path)
            zipf.write(abs_path, rel_path)

    zipf.close()


def zip_score_list(exp_list, savedir_base, out_fname, include_list=None):
    for exp_dict in exp_list:
        exp_id = hash_dict(exp_dict)
        zipdir("%s/%s" % (savedir_base, exp_id), 
               out_fname, include_list=include_list)