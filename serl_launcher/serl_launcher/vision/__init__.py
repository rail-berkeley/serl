from serl_launcher.vision.resnet_v1 import resnetv1_configs
from serl_launcher.vision.small_encoders import small_configs

encoders = dict()
encoders.update(resnetv1_configs)
encoders.update(small_configs)
