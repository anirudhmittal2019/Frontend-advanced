
import os


#pretrained_model_dir_path = os.path.abspath("./checkpoints")
pretrained_model_dir_path = os.path.abspath("")
cls_model_path = os.path.join(pretrained_model_dir_path, "classifier_model")

base_detect_model_path = os.path.join(
    pretrained_model_dir_path, "detector_v2_base_checkpoint")
base_detect_class_path = os.path.join(
    pretrained_model_dir_path, "detector_v2_base_classes")

default_detect_model_path = os.path.join(
    pretrained_model_dir_path, "detector_v2_default_checkpoint")
default_detect_class_path = os.path.join(
    pretrained_model_dir_path, "detector_v2_default_classes")


detect_model_path = os.path.join(pretrained_model_dir_path, "detector_model")