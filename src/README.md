### FishCount Repo
overleaf: https://www.overleaf.com/7594471487ndhphxdntznd
website:  https://alzayats.github.io/DeepFish/
related work: https://github.com/AlexOlsen/DeepWeeds
#### TODO

Timeline
- FishClf:
  - always-0, always-1 (run test_baselines.py)
  - ResNet, 1T-Hydra (run test.py -e resnet_clf deepfish_clf)

- FishReg:
  - ResNet, 1T-Hydra_R, 1T-Hydra_L, 2T-Hydra_RL, 3T-Hdra_CRL  (run test.py -e reg loc)

- FishLoc:
  - 1T-Hydra_L,  2T-Hydra_RL, 3T-Hdra_CRL  (run test.py -e loc)
  - 
- FishSeg:
  - 1T-Hydra_S,  2T-Hydra_LS, 3T-Hdra_RLS, 4T-Hdra_CRLS  (run test.py -e seg)

[1] http://openaccess.thecvf.com/content_cvpr_2017/papers/Chattopadhyay_Counting_Everyday_Objects_CVPR_2017_paper.pdf

#### 1. Running an experiment
The command below runs the `baseline` experiment set using the data and the save
directories of `issam`. 
```
python trainval.py -e baseline -u issam
```
The command below runs the `fish` experiment set using the data and the save
directories of `issam`. 

Download the full fish data from [here](https://cloudstor.aarnet.edu.au/plus/s/FDRH0b9NfzHks7e/download?path=%2FAlzayat&files=Public_JCU_Fish.zip)
```
python trainval.py -e fish -u issam
python trainval.py -e fish_glance -u alzayat
python trainval.py -e fish_lcfcn -u alzayat
```

You can reset the experiment with the command,
```
python trainval.py -e baseline -u issam -r 1
```

#### 2. Visualizing the Results
The following command outputs a dataframe and a plot 
that compare the results of the experiments in `baseline` 
using the save directory of `issam`.

```
python trainval.py  -e baseline -u issam -v epoch val_mae 
```

