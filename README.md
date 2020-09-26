

# DeepFish (A Realistic Fish-Habitat Dataset to Evaluate Algorithms for Underwater Visual Analysis) [Paper](https://www.nature.com/articles/s41598-020-71639-x)   
[![DOI](https://zenodo.org/badge/206528410.svg)](https://zenodo.org/badge/latestdoi/206528410)

### Accepted at Scientific Reports (Nature)

This repository contains the code to reproduce the experiments of the paper.
*  DeepFish  [Paper](https://www.nature.com/articles/s41598-020-71639-x).
*  Download the DeepFish dataset from [here](https://cloudstor.aarnet.edu.au/plus/s/NfjObIhtUYO6332)

![counting](count.gif) 
![Segmentation](seg.gif) 


### Reproducing paper experiments
![CNN](docs/Figure_4.png)



#### Installation
Download the repository:

`git clone https://github.com/alzayats/DeepFish.git`

Experiment hyperparameters are defined in `exp_configs.py`

Run the following command to reproduce the experiments in the paper:

`python trainval.py -e ${TASK} -sb ${SAVEDIR_BASE} -d ${DATADIR} -r 1`

The variables (`${...}`) can be substituted with the following values:

* `TASK` : clf, reg, loc, seg
* `SAVEDIR_BASE`: Absolute path to where results will be saved
* `DATADIR`: Absolute path containing the downloaded datasets



### Citations

If you use the DeepFish dataset in your work, please cite it as:

```
@article{saleh2020realistic,
    title={A Realistic Fish-Habitat Dataset to Evaluate Algorithms for Underwater Visual Analysis},
    author={Alzayat Saleh and Issam H. Laradji and Dmitry A. Konovalov and Michael Bradley and David Vazquez and Marcus Sheaves},
    year={2020},
    eprint={2008.12603},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
