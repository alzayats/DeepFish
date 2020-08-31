

# DeepFish (A Realistic Fish-Habitat Dataset to Evaluate Algorithms for Underwater Visual Analysis) [Paper](https://arxiv.org/abs/2008.12603)   
[![DOI](https://zenodo.org/badge/206528410.svg)](https://zenodo.org/badge/latestdoi/206528410)
### Accepted at Scientific Reports (Nature)

This repository contains the code to reproduce the experiments of the paper.
*  DeepFish  [Paper](https://arxiv.org/abs/2008.12603).
*  DeepFish Dataset [website](https://alzayats.github.io/DeepFish/).
*  Download the DeepFish dataset from [here](https://cloudstor.aarnet.edu.au/plus/s/NfjObIhtUYO6332)

![counting](count.gif) 
![Segmentation](seg.gif) 




### Reproducing paper experiments
![CNN](Figure_4.png)



Experiment hyperparameters are defined in `./exp_configs.py`

Run the following command to reproduce the experiments in the paper:

`python trainval.py -e ${TASK} -sb ${SAVEDIR_BASE} -d ${DATADIR} -r 1`

The variables (`${...}`) can be substituted with the following values:

* `TASK` : clf, reg, loc, seg
* `SAVEDIR_BASE`: Absolute path to where results will be saved
* `DATADIR`: Absolute path containing the downloaded datasets



### Citations

If you use the DeepFish dataset in your work, please cite it as:

IEEE style citation: “Alzayat Saleh, Issam H. Laradji, Dmitry A. Konovalov, David Vazquez, MichaelBradley, and Marcus Sheaves, 
“A Realistic Fish-Habitat Dataset to Evaluate Algorithms for Underwater Visual Analysis” *Scientific Reports*,............ ”

### BibTeX
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
