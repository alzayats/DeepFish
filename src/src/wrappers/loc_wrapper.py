import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import os
import numpy as np
import time
from src import utils as ut
from sklearn.metrics import confusion_matrix
import skimage
from src import wrappers
from src import mlkit
from skimage import morphology as morph
from skimage.morphology import watershed
from skimage.segmentation import find_boundaries
from scipy import ndimage


class LocWrapper(torch.nn.Module):
    def __init__(self, model, opt):
        super().__init__()
        self.model = model
        self.opt = opt

    def train_on_loader(self, train_loader):
        return wrappers.train_on_loader(self, train_loader)

    def val_on_loader(self, val_loader):
        val_monitor = LocMonitor()
        return wrappers.val_on_loader(self, val_loader, val_monitor=val_monitor)

    def vis_on_loader(self, vis_loader, savedir):
        return wrappers.vis_on_loader(self, vis_loader, savedir=savedir)

    def train_on_batch(self, batch, **extras):
        self.opt.zero_grad()
        self.train()

        images = batch["images"].cuda()
        counts = batch["counts"].cuda()

        blobs = self.predict_on_batch(batch, method="blobs").squeeze()

        points = batch["points"].long().cuda()

        points_numpy = mlkit.t2n(points).squeeze()
        blob_dict = get_blob_dict_base(self, blobs, points_numpy)

        self.train()
        loss = lc_loss_base(self.model.forward(images), images, points,
                            counts[None], blob_dict)


        loss.backward()

        self.opt.step()

        return {"loss_loc":loss.item()}

    def val_on_batch(self, batch, **extras):
        pred_counts = self.predict_on_batch(batch, method="counts") 
        pred_blobs = self.predict_on_batch(batch, method="blobs")

        val_dict = {}
        val_dict["val_loc"] = abs(pred_counts.ravel() - batch["counts"].numpy().ravel())
            
        pred_points = blobs2points(pred_blobs[0,0])
        gt_points = batch["points"].numpy()
        game_mean = 0.
        for L in range(4):
            game_mean += compute_game(pred_points, gt_points, L=L)

        val_dict["val_game"] = np.array([game_mean / (L + 1)])

        return val_dict
        
    def predict_on_batch(self, batch, **options):
        self.eval()
        # feat_8s, feat_16s, feat_32s = self.model.extract_features(batch["images"].cuda())

        if options["method"] == "counts":
            images = batch["images"].cuda()
            pred_mask = self.model.forward(images).data.max(1)[1].squeeze().cpu().numpy()

            counts = np.zeros(self.model.n_classes - 1)

            for category_id in np.unique(pred_mask):
                if category_id == 0:
                    continue
                blobs_category = morph.label(pred_mask == category_id)
                n_blobs = (np.unique(blobs_category) != 0).sum()
                counts[category_id - 1] = n_blobs

            return counts[None]

        elif options["method"] == "blobs":
            images = batch["images"].cuda()
            pred_mask = self.model.forward(images).data.max(1)[1].squeeze().cpu().numpy()

            h, w = pred_mask.shape
            blobs = np.zeros((self.model.n_classes - 1, h, w), int)

            for category_id in np.unique(pred_mask):
                if category_id == 0:
                    continue
                blobs[category_id - 1] = morph.label(pred_mask == category_id)

            return blobs[None]

        elif options["method"] == "points":
            images = batch["images"].cuda()
            pred_mask = self.model.forward(images).data.max(1)[1].squeeze().cpu().numpy()

            h, w = pred_mask.shape
            blobs = np.zeros((self.model.n_classes - 1, h, w), int)

            for category_id in np.unique(pred_mask):
                if category_id == 0:
                    continue
                blobs[category_id - 1] = morph.label(pred_mask == category_id)

            return blobs[None]
        
    def vis_on_batch(self, batch, savedir):
        from skimage.segmentation import mark_boundaries
        from skimage import data, io, segmentation, color
        from skimage.measure import label

        self.eval()
        pred_counts = self.predict_on_batch(batch, method="counts") 
        pred_blobs = self.predict_on_batch(batch, method="blobs")

        # loc 
        pred_count = pred_counts.ravel()[0]
        pred_blobs = pred_blobs.squeeze()

        img = mlkit.get_image(batch["images"],denorm="rgb")
        # img_points = mlkit.get_image(batch["images"],mask=batch["points"].squeeze(), enlarge=1, denorm="rgb")
        points = mlkit.zoom(batch["points"],11).squeeze()
        img_np = mlkit.f2l(img).squeeze().clip(0,1)

        out = color.label2rgb(label(pred_blobs), image=(img_np), image_alpha=1.0, bg_label=0)
        img_mask = mark_boundaries(out.squeeze(),  label(pred_blobs).squeeze())

        out = color.label2rgb(label(points), image=(img_np), image_alpha=1.0, bg_label=0)
        img_points = mark_boundaries(out.squeeze(),  label(points).squeeze())
        # img = 255*mlkit.get_image(batch["images"],mask=batch["points"].squeeze(), enlarge=1, denorm="rgb")
        # img = im.get_image(0.7*batch["images"] + 0.3*torch.FloatTensor(pred_blobs).squeeze(), denorm="rgb")
        # img_mask = 0.7*img[0] + 0.3*mlkit.l2f(mlkit.gray2cmap(pred_blobs)[0])
        # if batch["meta"]["index"] == 6:
        #     print(3)
        mlkit.save_image(savedir+"/images/%d.jpg" % batch["meta"]["index"], img_mask)
        # mlkit.save_image(savedir+"/images/%d_org.jpg" % batch["meta"]["index"], img)
        # mlkit.save_image(savedir+"/images/%d_gt.jpg" % batch["meta"]["index"], img_points)
        # mlkit.save_json(savedir+"/images/%d.json" % batch["meta"]["index"], 
        #             {"pred_count":float(pred_count), "gt_count": float(batch["counts"])})


    
class GAME:
    def __init__(self):
        super().__init__(higher_is_better=False)

        self.sum = 0.
        self.n_batches = 0.

        self.metric_name = type(self).__name__
        self.score_dict = {"metric_name": type(self).__name__}
        
        self.game_dict = {}
        for L in range(4):
            self.game_dict[L] = 0.

        self.game = 0.

    def add_batch(self, model, batch, **options):
        pred_blobs = ms.t2n(model.predict(batch, method="blobs")).squeeze()
        assert pred_blobs.ndim == 2
        pred_points = blobs2points(pred_blobs).squeeze()
        gt_points = ms.t2n(batch["points"]).squeeze()

        game_mean = 0.
        for L in range(4):
            game_sum = compute_game(pred_points, gt_points, L=L)
            self.game_dict[L] += game_sum
            game_mean += game_sum

        self.game += game_mean / (L + 1)
        self.n_batches += 1.


    def get_score_dict(self):
        curr_score = self.game/self.n_batches
        self.score_dict["score"] = curr_score

        # The Rest
        for L in range(4):
            self.score_dict["GAME%d"%L] = self.game_dict[L]/self.n_batches
        
        return self.score_dict


# -----------------------
# Utils
def blobs2points(blobs):
    points = np.zeros(blobs.shape).astype("uint8")
    rps = skimage.measure.regionprops(blobs)

    assert points.ndim == 2

    for r in rps:
        y, x = r.centroid

        points[int(y), int(x)] = 1


    # assert points.sum() == (np.unique(blobs) != 0).sum()
       
    return points

def compute_game(pred_points, gt_points, L=1):
    n_rows = 2**L
    n_cols = 2**L

    pred_points = pred_points.astype(float)
    gt_points = gt_points.astype(float)
    h, w = pred_points.shape
    se = 0.

    hs, ws = h//n_rows, w//n_cols
    for i in range(n_rows):
        for j in range(n_cols):

            sr, er = hs*i, hs*(i+1)
            sc, ec = ws*j, ws*(j+1)

            pred_count = pred_points[sr:er, sc:ec]
            gt_count = gt_points[sr:er, sc:ec]
            
            se += float(abs(gt_count.sum() - pred_count.sum()))
    return se



# ==========================================================
# Losses
def lc_loss(model, batch):
    model.train()

    blob_dict = get_blob_dict(model, batch)
    # put variables in cuda
    images = batch["images"].cuda()
    points = batch["points"].cuda()
    counts = batch["counts"].cuda()

    return lc_loss_base(model, images, points, counts, blob_dict)
    # print(images.shape)


def lc_loss_base(logits, images, points, counts, blob_dict):
    N = images.size(0)
    assert N == 1

    S = F.softmax(logits, 1)
    S_log = F.log_softmax(logits, 1)

    # IMAGE LOSS
    loss = compute_image_loss(S, counts)

    # POINT LOSS
    loss += F.nll_loss(S_log, points,
                       ignore_index=0,
                       reduction='sum')
    # FP loss
    if blob_dict["n_fp"] > 0:
        loss += compute_fp_loss(S_log, blob_dict)

    # split_mode loss
    if blob_dict["n_multi"] > 0:
        loss += compute_split_loss(S_log, S, points, blob_dict)

    # Global loss
    S_npy = mlkit.t2n(S.squeeze())
    points_npy = mlkit.t2n(points).squeeze()
    for l in range(1, S.shape[1]):
        points_class = (points_npy == l).astype(int)

        if points_class.sum() == 0:
            continue

        T = watersplit(S_npy[l], points_class)
        T = 1 - T
        scale = float(counts.sum())
        loss += float(scale) * F.nll_loss(S_log, torch.LongTensor(T).cuda()[None],
                                          ignore_index=1, reduction='mean')

    return loss / N


# Loss Utils
def compute_image_loss(S, Counts):
    n, k, h, w = S.size()

    # GET TARGET
    ones = torch.ones(Counts.size(0), 1).long().cuda()
    BgFgCounts = torch.cat([ones.float(), Counts.float()], 1)
    Target = (BgFgCounts.view(n * k).view(-1) > 0).view(-1).float()

    # GET INPUT
    Smax = S.view(n, k, h * w).max(2)[0].view(-1)

    loss = F.binary_cross_entropy(Smax, Target, reduction='sum')

    return loss


def compute_fp_loss(S_log, blob_dict):
    blobs = blob_dict["blobs"]

    scale = 1.
    loss = 0.
    n_terms = 0
    for b in blob_dict["blobList"]:
        if n_terms > 25:
            break

        if b["n_points"] != 0:
            continue

        T = np.ones(blobs.shape[-2:])
        T[blobs[b["class"]] == b["label"]] = 0

        loss += scale * F.nll_loss(S_log, torch.LongTensor(T).cuda()[None],
                                   ignore_index=1, reduction='mean')

        n_terms += 1
    return loss


def compute_bg_loss(S_log, bg_mask):
    loss = F.nll_loss(S_log, torch.LongTensor(bg_mask).cuda()[None],
                      ignore_index=1, reduction='mean')
    return loss


def compute_split_loss(S_log, S, points, blob_dict):
    blobs = blob_dict["blobs"]
    S_numpy = mlkit.t2n(S[0])
    points_numpy = mlkit.t2n(points).squeeze()

    loss = 0.

    for b in blob_dict["blobList"]:
        if b["n_points"] < 2:
            continue

        l = b["class"] + 1
        probs = S_numpy[b["class"] + 1]

        points_class = (points_numpy == l).astype("int")
        blob_ind = blobs[b["class"]] == b["label"]

        T = watersplit(probs, points_class * blob_ind) * blob_ind
        T = 1 - T

        scale = b["n_points"] + 1
        loss += float(scale) * F.nll_loss(S_log, torch.LongTensor(T).cuda()[None],
                                          ignore_index=1, reduction='mean')

    return loss


def watersplit(_probs, _points):
    points = _points.copy()

    points[points != 0] = np.arange(1, points.sum() + 1)
    points = points.astype(float)

    probs = ndimage.black_tophat(_probs.copy(), 7)
    seg = watershed(probs, points)

    return find_boundaries(seg)


@torch.no_grad()
def get_blob_dict(model, batch, training=False):
    blobs = model.predict(batch, method="blobs").squeeze()
    points = mlkit.t2n(batch["points"]).squeeze()

    return get_blob_dict_base(model, blobs, points, training=training)


def get_blob_dict_base(model, blobs, points, training=False):
    if blobs.ndim == 2:
        blobs = blobs[None]

    blobList = []

    n_multi = 0
    n_single = 0
    n_fp = 0
    total_size = 0

    for l in range(blobs.shape[0]):
        class_blobs = blobs[l]
        points_mask = points == (l + 1)
        # Intersecting
        blob_uniques, blob_counts = np.unique(class_blobs * (points_mask), return_counts=True)
        uniques = np.delete(np.unique(class_blobs), blob_uniques)

        for u in uniques:
            blobList += [{"class": l, "label": u, "n_points": 0, "size": 0,
                          "pointsList": []}]
            n_fp += 1

        for i, u in enumerate(blob_uniques):
            if u == 0:
                continue

            pointsList = []
            blob_ind = class_blobs == u

            locs = np.where(blob_ind * (points_mask))

            for j in range(locs[0].shape[0]):
                pointsList += [{"y": locs[0][j], "x": locs[1][j]}]

            assert len(pointsList) == blob_counts[i]

            if blob_counts[i] == 1:
                n_single += 1

            else:
                n_multi += 1
            size = blob_ind.sum()
            total_size += size
            blobList += [{"class": l, "size": size,
                          "label": u, "n_points": blob_counts[i],
                          "pointsList": pointsList}]

    blob_dict = {"blobs": blobs, "blobList": blobList,
                 "n_fp": n_fp,
                 "n_single": n_single,
                 "n_multi": n_multi,
                 "total_size": total_size}

    return blob_dict



class LocMonitor:
    def __init__(self):
        self.ae = 0
        self.ae_game = 0
        self.n_samples = 0

    def add(self, val_dict):
        self.ae += val_dict["val_loc"].sum()
        self.ae_game += val_dict["val_game"].sum()
        self.n_samples += val_dict["val_loc"].shape[0]

    def get_avg_score(self):
        return {"val_loc":self.ae/ self.n_samples, 
              "val_game":self.ae_game/ self.n_samples}


