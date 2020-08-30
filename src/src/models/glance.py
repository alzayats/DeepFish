from torch import nn
from torchvision import models
from torch.autograd import Variable
# from haven.base import base_model
import torch.nn.functional as F
import torch
from torch import optim
import torchvision
# from haven._toolbox import misc as ms
import numpy as np
import shutil
from src import utils as ut
from src import models as md
# from models.counts2points import helpers
from torchvision.transforms import functional as FT
from torch.autograd import Function
from . import resfcn


class Glance_Res50(torch.nn.Module):
    def __init__(self, n_classes=2, layers="100-100"):
        layers = list(map(int, str(layers).split("-")))
        
        n_outputs = 1

        super().__init__()

        self.backbone = resfcn.ResBackbone()
        # model = models.vgg16(pretrained=True)
        # self.features = model.features
        # self.fc7 = nn.Sequential(
        #     # stop at last layer
        #     *list(model.classifier.children())[:-1]
        # )

        # Freeze the pretrained-weights
        # for p in self.features.parameters():
        #     p.requires_grad = False

        # for p in self.fc7.parameters():
        #     p.requires_grad = False        

        # CREATE INTERMEDIATE MLP LAYERS
        layers = [98] + layers
        n_hidden = len(layers) - 1

        layerList = []      
        for i in range(n_hidden): 
            layerList += [nn.Linear(layers[i], 
                          layers[i+1]), nn.ReLU()]
        
        layerList += [nn.Linear(layers[i+1], 
                        n_outputs)]
        self.mlp = nn.Sequential(*layerList)


    def forward(self, x):
        n = x.shape[0]
        logits_32s, logits_16s, logits_8s = self.backbone.extract_features(x)

        # 1. EXTRACT resnet features
        x = logits_32s.view(n, -1)
       
        
        # 2. GET MLP OUTPUT
        x = self.mlp(x)
        return x 

    def predict(self, batch, metric="counts"):
        self.eval()

        masks = batch["images"].cuda()
        pred = self.forward(masks)

        return pred.data.round()


class Glance_VGG16(nn.Module):
    def __init__(self, init=True):
        super().__init__()
        if init:
            layers = "500-500"
            layers = list(map(int, str(layers).split("-")))

            n_outputs = 1
            super().__init__()

            model = models.vgg16(pretrained=True)
            self.features = model.features
            self.fc7 = nn.Sequential(
                # stop at last layer
                *list(model.classifier.children())[:-1]
            )

            # CREATE INTERMEDIATE MLP LAYERS
            layers = [4096] + layers
            n_hidden = len(layers) - 1

            layerList = []
            for i in range(n_hidden):
                layerList += [nn.Linear(layers[i],
                                        layers[i + 1]), nn.ReLU()]

            layerList += [nn.Linear(layers[i + 1],
                                    n_outputs)]

            self.mlp = nn.Sequential(*layerList)

            self.opt = optim.Adam(
                self.parameters(), lr=1e-5,
                weight_decay=0.0005)

    def forward(self, images):

        # 1. EXTRACT vgg16 features
        x = self.features(images)
        x = x.view(x.size(0), -1)
        x = self.fc7(x)

        # 2. GET MLP OUTPUT
        x = self.mlp(x)
        return x

    def train_step(self, batch):
        self.train()
        images = batch["images"].cuda()
        counts = batch["counts"].cuda()
        self.opt.zero_grad()

        pred_counts = self(images)

        loss = F.mse_loss(pred_counts.squeeze(), counts.float().squeeze(), reduction="mean")
        loss.backward()
        self.opt.step()

        return loss.item()


    @torch.no_grad()
    def predict(self, batch, **options):
        self.eval()

        if options.get("method") == "points":
            pred_dict = self.predict(batch, method="pred_dict")

            return hut.t2n(pred_dict["points"])

        elif options.get("method") == "pred_dict":
            gray, pos, neg = self.predict(batch, method="sal")
            # if not hasattr(self, "proposal"):
            #     self.proposal = md.get_model_class("RPN")(self.exp_dict).cuda()

            # get proposals
            # proposal_dict = self.exp_dict.get("proposal_dict") or {}
            batch["image_pil"] = FT.to_pil_image(batch["images_original_tensor"][0])
            annList = helpers.get_rpn_annList(self.proposal, batch,
                                                counts = batch["counts"],
                                              **proposal_dict)

            # score them
            for i in range(len(annList)):
                annList[i]["score"] = 0.
            annList_scored = helpers.score_annList(annList,
                                                   probs=gray+pos+neg,
                                                   alpha=1.0)

            # select best
            annList_best = helpers.select_best(annList, counts=batch["counts"], method="C_WSL")

            # get points and mask
            if len(annList_best) == 0:
                _, h, w = batch["points"].shape
                points = np.zeros((h,w))
                mask = np.zeros((h, w))
            else:
                points, mask = helpers.get_points_masks(annList_best)

            return {"annList":annList,
                    "annList_scored":annList_scored,
                    "annList_best":annList_best,
                    "points":points,
                    "mask":mask, "gray":gray,
                    "pos":pos, "neg":neg}

        elif options.get("method") == "sal":
            gradient = 1
            sal = guided_saliency(self,
                                  batch["images"],
                                  gradient=gradient)[0]
            gray = convert_to_grayscale(sal)
            pos, neg = get_positive_negative_saliency(sal)
            # pos, neg = pos.mean(0), neg.mean(0)
            pos = convert_to_grayscale(pos)
            neg = convert_to_grayscale(neg)

            return gray, pos, neg


        elif options.get("method") == "counts":
            pred_counts = self.forward(batch["images"].cuda())
            counts = pred_counts.round()
            counts = hut.t2n(counts)
            return counts.squeeze()
            # return hut.t2n(counts)

    def visualize(self, batch, savedir, **options):
        index = batch["meta"]["index"].item()

        path_base = "%s/%d" % (savedir, index)

        pred_counts = self.predict(batch, method="counts").item()
        gt_counts = batch["counts"].item()

        name = "preds"
        img1 = ms.get_image(
                            np.array(batch["images_original"]),
                            # mask=blobs.squeeze()
                            )
        meta_dict = {"pred_counts": pred_counts, "gt_counts": gt_counts, "name": name}
        ut.save_img_pkl(img1, meta_dict, path_base, name=name)
        ###################
        pred_dict = self.predict(batch, method="pred_dict")
        if 1:

            gray, pos, neg = pred_dict["gray"], pred_dict["pos"], pred_dict["neg"]
            # gray[gray < 0.5] = 0
            # ===============================
            #
            name = "gray"
            img1 = ms.get_image(np.array(batch["images_original"]) * 0.4
                                + 0.6 * ms.t2n(ms.gray2cmap(gray)))

            meta_dict = {"pred": pred_counts,
                         "counts": gt_counts,
                         "name": name}
            ut.save_img_pkl(img1, meta_dict,
                            path_base, name=name)
            # ===============================
            #
            # name = "pos"
            # img1 = ms.get_image(np.array(batch["images_original"]) * 0.4
            #                     + 0.6 * ms.t2n(ms.gray2cmap(pos)))
            #
            # meta_dict = {"pred": pred_counts,
            #              "counts": gt_counts,
            #              "name": name}
            # ut.save_img_pkl(img1, meta_dict,
            #                 path_base, name=name)

            # ===============================
            #
            # name = "neg"
            # img1 = ms.get_image(np.array(batch["images_original"]) * 0.4
            #                     + 0.6 * ms.t2n(ms.gray2cmap(neg)))
            #
            # meta_dict = {"pred": pred_counts,
            #              "counts": gt_counts,
            #              "name": name}
            # ut.save_img_pkl(img1, meta_dict,
            #                 path_base, name=name)
            # ===============================
            #

            name = "annList_best"
            annList_best = pred_dict[name]
            img1 = ms.get_image(np.array(batch["images_original_tensor"]), annList=annList_best)
            meta_dict = {"index": index, "name": name, "n_proposals": len(annList_best)}
            ut.save_img_pkl(img1, meta_dict, path_base, name=name)

            # ===============================
            #
            name = "pred_points"
            img1 = ms.get_image(np.array(batch["images_original_tensor"]), mask=pred_dict["points"].squeeze(), enlarge=1)

            meta_dict = {"pred": pred_counts,
                         "counts": gt_counts,
                         "name": name}
            ut.save_img_pkl(img1, meta_dict,
                            path_base, name=name)

            # ===============================
            #
            # name = "gt_points"
            # img1 = ms.get_image(np.array(batch["images_original_tensor"]).squeeze(),
            #                     mask=batch["points"].squeeze(), enlarge=1)
            #
            # meta_dict = {"pred": pred_counts,
            #              "counts": gt_counts,
            #              "name": name}
            # ut.save_img_pkl(img1, meta_dict,
            #                 path_base, name=name)


class Glance_PRM(Glance_VGG16):
    def __init__(self):
        super().__init__(init=False)
        self.with_counts = False
        num_classes = 1
        model = torchvision.models.resnet50(pretrained=True)
        # feature encoding
        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4)

        # classifier
        num_features = model.layer4[1].conv1.in_channels
        self.classifier = nn.Sequential(
            nn.Conv2d(num_features, num_classes, kernel_size=1, bias=True))

        # # FREEZE BATCH NORMS
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.requires_grad = False
                m.bias.requires_grad = False

        self.opt = optim.Adam(
            self.parameters(), lr=1e-5,
            weight_decay=0.0005)


    def forward(self, x, counts=None):
        self.features.eval()
        x = self.features(x)
        class_response_maps = self.classifier(x)
        self.sub_pixel_locating_factor = 8
        self.win_size, self.peak_filter = 3, _median_filter

        class_response_maps = F.interpolate(class_response_maps,
                                            scale_factor=self.sub_pixel_locating_factor,
                                            mode='bilinear', align_corners=True)
        # aggregate responses from informative receptive fields estimated via class peak responses
        peak_list, aggregation = peak_stimulation(class_response_maps,
                                                  win_size=self.win_size,
                                                  peak_filter=self.peak_filter,
                                                  counts=counts)

        return aggregation

    def train_step(self, batch):
        self.train()
        images = batch["images"].cuda()
        counts = batch["counts"].cuda()

        self.opt.zero_grad()

        if self.with_counts:
            pred_counts = self(images, counts=True)
        else:
            pred_counts = self(images)
        loss = F.mse_loss(pred_counts.squeeze() ,counts.float().squeeze())
        loss.backward()

        self.opt.step()

        return loss.item()

        # loss = F.mse_loss(pred_counts.squeeze(), counts.float().squeeze(), reduction="mean")
        # loss.backward()
        #
        # return loss.item()
    #
    # def visualize(self, batch, savedir, **options):
    #     pass
