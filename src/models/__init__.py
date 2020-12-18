from . import lcfcn, csrnet, glance
from . import resnet, fcn8, inception, unet, fcn8_vgg16



def get_model(model_name, exp_dict):
    if model_name == "fcn8":
        model = fcn8.FCN8()
    
    if model_name == "fcn8_vgg16":
        model = fcn8_vgg16.FCN8VGG16(n_classes=1)

    if model_name == "unet":
        model = unet.UNET()

    if model_name == "resnet":
        model = resnet.ResNet(n_classes=1)

    if model_name == "inception":
        model = inception.inceptionresnetv2(num_classes=1)

    return model
