from src import mlkit

EXP_GROUPS = {
  "fisheries": {"dataset":[
        # "tiny_fish_loc", 
        # "fish_loc"
        "Como_Lake_2",
        "Chimney",
        "Green_Timbers",
        "Hastings",
        "kentucky",
        "lafrage",
        "yellow_docks_1",
        "Rice_lake"
        ],
                      "task": ["loc"],
                        "model":[
                           "fcn8"],
                        "batch_size": [1],
                        "transform":["rgb_normalize"],
                        "max_epoch": [1000],
                        "wrapper":["loc_wrapper"]},

     "clf": {"dataset":[
      #  "tiny_fish_clf", 
      "fish_clf"],
            "task":["clf"],
            "model":[ 
              "inception", 
            "resnet"],
            "batch_size": [1],
            "transform":["resize_normalize"],
            "max_epoch": [1000],
            "wrapper":["clf_wrapper"]},

     "reg": {"dataset":[
      #  "tiny_fish_reg",
       "fish_reg"],
                            "task":["reg"],
                           "model":[
                             "inception",
                            "resnet"],
                           "batch_size": [1],
                           "transform":["resize_normalize"],
                           "max_epoch": [1000],
                           "wrapper":["reg_wrapper"]},

      "loc": {"dataset":[
        # "tiny_fish_loc", 
        "fish_loc"],
                      "task": ["loc"],
                        "model":[
                          "unet",
                           "fcn8"],
                        "batch_size": [1],
                        "transform":["rgb_normalize"],
                        "max_epoch": [1000],
                        "wrapper":["loc_wrapper"]},


      "seg": {"dataset":[
        # "tiny_fish_seg", 
        "fish_seg"],
                      "task": ["seg"],
                        "model":[
                          "unet",
                           "fcn8"],
                        "batch_size": [1],
                        "transform":["rgb_normalize"],
                        "max_epoch": [1000],
                        "wrapper":["seg_wrapper"]},
             }


EXP_GROUPS = {k:mlkit.cartesian_exp_group(v) for k,v in EXP_GROUPS.items()}