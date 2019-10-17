import pretrainedmodels
import torch.nn as nn
from torchvision.models import resnet18, resnet34


# ['fbresnet152', 'bninception', 'resnext101_32x4d', 'resnext101_64x4d', 'inceptionv4', 'inceptionresnetv2',
# 'alexnet', 'densenet121', 'densenet169', 'densenet201', 'densenet161', 'resnet18', 'resnet34', 'resnet50',
# 'resnet101', 'resnet152', 'inceptionv3', 'squeezenet1_0', 'squeezenet1_1', 'vgg11', 'vgg11_bn', 'vgg13',
# 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19_bn', 'vgg19', 'nasnetamobile', 'nasnetalarge', 'dpn68', 'dpn68b', 'dpn92',
# 'dpn98', 'dpn131', 'dpn107', 'xception', 'senet154', 'se_resnet50', 'se_resnet101', 'se_resnet152',
# 'se_resnext50_32x4d', 'se_resnext101_32x4d', 'cafferesnet101', 'pnasnet5large', 'polynet']


def get_model(config, num_classes=196):
    model_name = config.MODEL.NAME

    if model_name.startswith('resnet'):
        model = globals().get(model_name)(pretrained=True)
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
        model.avg_pool = nn.AdaptiveAvgPool2d(1)
        in_features = model.last_linear.in_features
        model.last_linear = nn.Linear(in_features, num_classes)
    print('model name:', model_name)

    if config.MODEL.ADD_FC:
        new_fc = nn.Sequential(
                        nn.Linear(in_features, 512),
                        nn.BatchNorm1d(512),
                        nn.ReLU(),
                        nn.Dropout(0.5),
                        nn.Linear(512, num_classes))
        if model_name.startswith('resnet'):
            model.fc = new_fc
        else:
            model.last_linear = new_fc
        print('fc added')

    if config.PARALLEL:
        model = nn.DataParallel(model)

    return model

