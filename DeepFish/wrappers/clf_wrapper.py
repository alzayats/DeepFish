import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import os
import numpy as np
import time
from DeepFish import utils as ut
from sklearn.metrics import confusion_matrix
import skimage
from DeepFish import wrappers
from torchvision import transforms
from haven import haven_utils as hu


class ClfWrapper(torch.nn.Module):
    def __init__(self, model, opt):
        super().__init__()
        self.model = model
        self.opt = opt

    def train_on_loader(self, train_loader):
        return wrappers.train_on_loader(self, train_loader)

    def val_on_loader(self, val_loader):
        val_monitor = ClfMonitor()
        return wrappers.val_on_loader(self, val_loader, val_monitor=val_monitor)

    def vis_on_loader(self, vis_loader, savedir):
        return wrappers.vis_on_loader(self, vis_loader, savedir=savedir)

    def train_on_batch(self, batch, **extras):
        self.opt.zero_grad()

        device = next(self.model.parameters()).device
        labels = batch["labels"].to(device)
        logits = self.model.forward(batch["images"].to(device))
        loss_clf =  F.binary_cross_entropy_with_logits(logits.squeeze(),
                        labels.squeeze().float(), reduction="mean")
        loss_clf.backward()

        self.opt.step()

        return {"loss_clf":loss_clf.item()}

    def val_on_batch(self, batch, **extras):
        pred_clf = self.predict_on_batch(batch)
        return (pred_clf.cpu().numpy().ravel() != batch["labels"].numpy().ravel())
        
    def predict_on_batch(self, batch):
        device = next(self.model.parameters()).device
        images = batch["images"].to(device)
        n = images.shape[0]
        logits = self.model.forward(images)
        return (torch.sigmoid(logits) > 0.5).float()
        
    def vis_on_batch(self, batch, savedir_image):
        self.eval()
        # clf
        pred_labels = float(self.predict_on_batch(batch))
        img = hu.get_image(batch["image_original"], denorm="rgb")
        img = np.array(img)
        hu.save_image(savedir_image+"/images/%d.jpg" % batch["meta"]["index"], img)
        hu.save_json(savedir_image+"/images/%d.json" % batch["meta"]["index"],
                    {"pred_label":float(pred_labels), "gt_label": float(batch["labels"])})

    @torch.no_grad()
    def vis_on_batch_inference(self, batch, savedir_image):
        """Visualize classification predictions without ground truth comparison.

        This method outputs the image and prediction without requiring ground truth labels.
        Returns both the binary prediction and the raw sigmoid probability (confidence).
        """
        self.eval()
        device = next(self.model.parameters()).device
        logits = self.model.forward(batch["images"].to(device))
        prob = float(torch.sigmoid(logits).item())
        pred_labels = float(prob > 0.5)

        # Try to get original image if available, otherwise use transformed image
        if "image_original" in batch:
            img = hu.get_image(batch["image_original"], denorm="rgb")
        else:
            img = hu.get_image(batch["images"], denorm="rgb")
        img = np.array(img)

        return {"prediction": pred_labels, "confidence": prob, "image": img}


class ClfMonitor:
    def __init__(self):
        self.corrects = 0
        self.n_samples = 0

    def add(self, corrects):
        self.corrects += corrects.sum()
        self.n_samples += corrects.shape[0]

    def get_avg_score(self):
        return {"val_clf": self.corrects/ self.n_samples}