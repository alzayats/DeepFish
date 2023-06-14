

## DeepFish (A Realistic Fish-Habitat Dataset to Evaluate Algorithms for Underwater Visual Analysis) 

### Accepted at Nature Scientific Reports [[Paper]](https://www.nature.com/articles/s41598-020-71639-x) [[Dataset]](http://data.qld.edu.au/public/Q5842/2020-AlzayatSaleh-00e364223a600e83bd9c3f5bcd91045-DeepFish/)  [[Project]](https://alzayats.github.io/DeepFish/) 

![CNN](docs/Figure_4.png)
![counting](docs/count.gif) 
![Segmentation](docs/seg.gif) 

The dataset consists of approximately 40 thousand images collected underwater from 20 habitats in the marine-environments of tropical Australia.
The dataset originally contained only classification labels. Thus, we collected point-level and segmentation labels to have a more comprehensive fish analysis benchmark.
Videos for DeepFish were collected for 20 habitats from remote coastal marine environments of tropical Australia. These videos were acquired using cameras mounted on metal frames, deployed over the side of a vessel to acquire video footage underwater. The cameras were lowered to the seabed and left to record the natural fish community, while the vessel maintained a distance of 100 m. The depth and the map coordinates of the cameras were collected using an acoustic depth sounder and a GPS, respectively. Video recording was carried out during daylight hours and in relatively low turbidity periods. The video clips were captured in full HD resolution (1920 × 1080 pixels) from a digital camera. In total, the number of video frames taken is 39,766. 
[[more]](https://research.jcu.edu.au/data/published/48fcdde6576ee929325b01fca4207914/)

## To install DeepFish as a Python package for access outside the repo:
`python setup.py install` OR `pip install -e .`

## Install requirements
`pip install -r requirements.txt` 

`pip install git+https://github.com/ElementAI/LCFCN`

## Download

*  Download the DeepFish dataset from [here](http://data.qld.edu.au/public/Q5842/2020-AlzayatSaleh-00e364223a600e83bd9c3f5bcd91045-DeepFish/)

## 1. Train and test on single image

### Localization
```
python scripts/train_single_image.py -e loc -d ${PATH_TO_DATASET}
```

This outputs the following image 

![CNN](docs/single_image_loc.png)

### Segmentation

```
python scripts/train_single_image.py -e seg -d ${PATH_TO_DATASET}
```

This outputs the following image 

![CNN](docs/single_image_seg.png)

## 2. Train and test on the dataset

Run the following command to reproduce the experiments in the paper:

`python trainval.py -e ${TASK} -sb ${SAVEDIR_BASE} -d ${DATADIR} -r 1`

The variables (`${...}`) can be substituted with the following values:

* `TASK` : loc, seg, clf, reg
* `SAVEDIR_BASE`: Absolute path to where results will be saved
* `DATADIR`: Absolute path containing the downloaded datasets

Experiment hyperparameters are defined in `exp_configs.py`

## Citations

If you use the DeepFish dataset in your work, please cite it as:

```
@article{saleh2020realistic,
  title={A realistic fish-habitat dataset to evaluate algorithms for underwater visual analysis},
  author={Saleh, Alzayat and Laradji, Issam H and Konovalov, Dmitry A and Bradley, Michael and Vazquez, David and Sheaves, Marcus},
  journal={Scientific Reports},
  volume={10},
  number={1},
  pages={14671},
  year={2020},
  publisher={Nature Publishing Group UK London},
  doi={https://doi.org/10.1038/s41598-020-71639-x}
}
```
