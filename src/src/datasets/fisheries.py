import time
from torch.utils import data
import glob
import datetime
import pandas as pd 
import scipy.misc as m
from bs4 import BeautifulSoup
import numpy as np
import torch
from torch.nn import functional as F
import pickle
import os
from torch.autograd import Variable
import json
from importlib import reload
from torch.utils import data
import glob
import datetime
import pandas as pd 
import scipy.misc as m
from bs4 import BeautifulSoup
import numpy as np
import torch
import imageio
from scipy.io import loadmat
import cv2
from skimage.segmentation import mark_boundaries
from tqdm import tqdm
from skimage.segmentation import slic
from torchvision import transforms
from PIL import Image
from src import mlkit
from scipy.ndimage.filters import gaussian_filter 
# from addons import transforms as myTransforms
# import misc as ms


class Fisheries(data.Dataset):
    def __init__(self, split, transform=None, datadir="", n_samples=None, lake=None):  
        self.datadir = "/mnt/projects/counting/Saves/DatasetsPrivate/fisheries/%s/" % lake 
        self.split = split
        # self.transform_name = transform_name
        corrupted = []
        self.transform = transform
        # 2. GET META
        mname = self.datadir + '/logs/meta_%s.pkl' % lake
        mlkit.create_dirs(mname)

        if os.path.exists(mname) and False:
            with open(mname, 'rb') as f:
                meta = pickle.load(f)
        else:
            meta = get_meta(self.datadir, corrupted)

            # read python dict back from the file
            with open(mname, 'wb') as f:
                pickle.dump(meta, f)

        n = len(meta)
        # 1. LAKE CONFIGURATIONS
        e_train = (n * 3)//8
        e_val = n // 2

        if split == "train":
          indices = np.arange(e_train) 
          
        if split == "val":
          indices = np.arange(e_train, e_val)

        elif split == "test":
          indices = np.arange(e_val, n)

        self.meta = np.array(meta)[indices]

        self.n_classes = 2

        self.n_objects = self.n_fishers = np.array(pd.DataFrame(list(self.meta))["n_fisher"])
        
    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index):
        img_meta = self.meta[index]
        img_path = self.datadir + img_meta["img"]
        try:
          image_pil = Image.open(img_path).convert('RGB')
        except:
          return self.__getitem__(np.random.choice(len(self)))
        w, h = image_pil.size
        image_pil = image_pil.resize((int(w*0.25), int(h*0.25)))
        w, h = image_pil.size
        image = self.transform(image_pil)

        # GET POINTS
        
        points = np.zeros((h, w), np.uint8)

        for p in img_meta["pointList"]:
            points[int(p["y"] * h), 
                   int(p["x"] * w)] = 1
        
        points = transforms.functional.to_pil_image(points[:,:,np.newaxis])
        points = torch.FloatTensor(np.array(points))

        # COMPUTE COUNTS
        
        counts = torch.LongTensor(np.array([int(points.sum())]))
        batch = {"images": image,
                 "counts": float(counts.item()),
                 "points": points,
                 "meta": {"index": index,
                          "image_id": index,
                          "split": self.split}}

        return batch

def read_xml(fname):
    with open(fname) as f:
        xml = f.readlines()
    xml = ''.join([line.strip('\t') for line in xml])
    
    xml = BeautifulSoup(xml, "lxml")

    return xml

def read_img(fname):
    pass

def get_meta(path, corrupted):
    xml = read_xml(path + "ImageData.xml")
    xml_meta = xml.findAll("image")

    meta = []
    for img in xml_meta:
        img_name = str(img.data.contents[0])

        if img_name in corrupted:
            continue

        points = img.findAll("point")
     
        pointList = []
        n_count = 0
        for p in points:
            xc = float(p.x.contents[0])
            yc = float(p.y.contents[0])

            pointList += [{"y":yc, "x":xc}]

            n_count += 1


        meta += [{
                  "img":img_name, 
                  "pointList":pointList,
                  "n_fisher":n_count}]
    return meta


class ComoLake(Fisheries):
    def __init__(self, root="", download=0, split=None, 
                 transform_name=None, 
                 ratio=0.1, density=0, sigma=8.0,Time=False):
        self.split = split
        lake = "Como_Lake_2"
        self.ratio = ratio

        
        self.sigma = sigma
        self.density = density


        corrupted = []    
        super(ComoLake, self).__init__(root,
                  lake=lake, 
                 transform_name=transform_name,
                 corrupted=corrupted,
                 split=split,Time=Time)

        
def find_corrupted(imgList):
  corrupted = []
  n= len(imgList)
  for j, img_path in enumerate(imgList):
    print(j, n)
    try:
      _img = Image.open(img_path).convert('RGB')
      
    except:
      corrupted += [ut.extract_fname(img_path)]

  return corrupted

class YellowDocks(Fisheries):
    def __init__(self, root="", download=0, split=None, 
                 transform_name=None, 
                 ratio=0.2, density=0, sigma=8.0, Time=False):

        lake = "yellow_docks_1"
        self.ratio = ratio

        self.sigma = sigma
        self.density = density
        self.split = split

        #corrupted = ["MDGC4877.JPG"]
        '''
          names = glob.glob(tmp_path + "/*.JPG")
          corrupted = find_corrupted(names )
          ms.save_pkl(tmp_path + "/corrupted.pkl", corrupted)
        '''
        tmp_path = "/mnt/home/issam/DatasetsPrivate/fisheries/yellow_docks_1/"
        corrupted = ms.load_pkl(tmp_path + "/corrupted.pkl")

        super(YellowDocks, self).__init__(root,
                 split=split,
                  lake=lake, 
                 transform_name=transform_name,
                 corrupted=corrupted,Time=Time)

        # GET CORRUPTED


class YellowDocks25(YellowDocks):
    def __init__(self, root="", download=0, split=None, 
                 transform_name=None, 
                 ratio=0.2, density=0, sigma=8.0, Time=False):

        super().__init__(root,
                 split=split,
                  transform_name=transform_name,
                  Time=Time)
        if split == "train":
          self.meta = self.meta[:int(len(self.meta)*0.25)]



class YellowDocks50(YellowDocks):
    def __init__(self, root="", download=0, split=None, 
                 transform_name=None, 
                 ratio=0.2, density=0, sigma=8.0, Time=False):

        super().__init__(root,
                 split=split,
                  transform_name=transform_name,
                  Time=Time)
        if split == "train":
          self.meta = self.meta[:int(len(self.meta)*0.50)]


class YellowDocks75(YellowDocks):
    def __init__(self, root="", download=0, split=None, 
                 transform_name=None, 
                 ratio=0.2, density=0, sigma=8.0, Time=False):

        super().__init__(root,
                 split=split,
                  transform_name=transform_name,
                  Time=Time)
        if split == "train":
          self.meta = self.meta[:int(len(self.meta)*0.75)]


 
class Lafrage(Fisheries):
    def __init__(self, root="", download=0, split=None, 
                 transform_name=None, 
                 ratio=0.2, density=0, sigma=8.0, Time=False):
        self.split = split
        self.ratio = ratio
        lake = "lafrage"
        
        self.sigma = sigma
        self.density = density


        corrupted = []    
        super(Lafrage, self).__init__(root,
                 split=split,
                  lake=lake, 
                 transform_name=transform_name,
                 corrupted=corrupted,Time=Time)

class RiceLake(Fisheries):
    def __init__(self, root="", download=0, split=None, 
                 transform_name=None, 
                 ratio=0.3, density=0, sigma=8.0, Time=False):
        self.split = split
        self.ratio = ratio
        lake = "Rice_lake"
        
 
 
        self.sigma = sigma
        self.density = density

        corrupted = []    
        super(RiceLake, self).__init__(root,
                 split=split,
                  lake=lake, 
                 transform_name=transform_name,
                 corrupted=corrupted,Time=Time)


class GreenTimbers(Fisheries):
    def __init__(self, root="", download=0, split=None, 
                 transform_name=None, 
                 ratio=0.2, density=0, sigma=8.0, Time=False):
        self.split = split
        self.ratio = ratio
        lake = "Green_Timbers"
        

        self.sigma = sigma
        self.density = density

        corrupted = []    
        super(GreenTimbers, self).__init__(root,
                 split=split,
                  lake=lake, 
                 transform_name=transform_name,
                 corrupted=corrupted,Time=Time)


class Chimney(Fisheries):
    def __init__(self, root="", download=0, split=None, 
                 transform_name=None, 
                 ratio=0.5, density=0, sigma=8.0, Time=False):
        self.split = split
        self.ratio = ratio
        lake = "Chimney"
        


        self.sigma = sigma
        self.density = density

        corrupted = []    
        super(Chimney, self).__init__(root,
                  split=split,
                  lake=lake, 
                 transform_name=transform_name,
                 corrupted=corrupted,Time=Time)


class Hastings(Fisheries):
    def __init__(self, root="", download=0, split=None, 
                 transform_name=None, 
                 ratio=0.3, density=0, sigma=8.0, Time=False):
        self.split = split
        self.ratio = ratio
        lake = "Hastings"
        



        self.sigma = sigma
        self.density = density

        corrupted = []    
        super(Hastings, self).__init__(root,
                  split=split,
                  lake=lake, 
                 transform_name=transform_name,
                 corrupted=corrupted,Time=Time)



class Kentucky(Fisheries):
    def __init__(self, root="", download=0, split=None, 
                 transform_name=None, 
                 ratio=0.3, density=0, sigma=8.0, Time=False):
        self.split = split
        self.ratio = ratio
        lake = "kentucky"
        

        self.sigma = sigma
        self.density = density
        
        corrupted = []    
        super(Kentucky, self).__init__(root,
                  split=split,
                  lake=lake, 
                 transform_name=transform_name,
                 corrupted=corrupted,Time=Time)