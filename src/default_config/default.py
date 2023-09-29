from yacs.config import CfgNode as CN

_CN = CN()

############### MODEL ###############
_CN.MODEL_TYPE = "dino"

_CN.MODEL = CN()

_CN.MODEL.N_PRETRAINED_BLOCKS = 11  # number of pretrained DINO blocks
_CN.MODEL.VIT_BACKBONE = "vitb8"

_CN.MODEL.FEATURE_DIM = 64
_CN.MODEL.POS_LANDMARK_RADIUS = 0.2
_CN.MODEL.NEG_LANDMARK_RADIUS = 10.0
_CN.MODEL.LANDMARK_EMBEDDING_METHOD = "sampled_patch"


############### DATA ###############
_CN.DATASET = CN()
_CN.DATASET.TRAIN_DATA_SOURCE = "Matterport"
_CN.DATASET.VAL_DATA_SOURCE = "Matterport"
_CN.DATASET.TEST_DATA_SOURCE = "Matterport"

_CN.DATASET.TRAIN_DATA_ROOT = None
_CN.DATASET.TRAIN_POSE_ROOT = None
_CN.DATASET.TRAIN_INTRINSICS_PATH = None
_CN.DATASET.TRAIN_SCENE_LIST = None  # file path for list of scene names

_CN.DATASET.VAL_DATA_ROOT = None
_CN.DATASET.VAL_POSE_ROOT = None
_CN.DATASET.VAL_INTRINSICS_PATH = None
_CN.DATASET.VAL_SCENE_LIST = None  # file path for list of scene names

_CN.DATASET.TEST_DATA_ROOT = None
_CN.DATASET.TEST_POSE_ROOT = None
_CN.DATASET.TEST_INTRINSICS_PATH = None
_CN.DATASET.TEST_SCENE_LIST = None  # file path for list of scene names

_CN.DATASET.IMG_HEIGHT = 256
_CN.DATASET.IMG_WIDTH = 320
_CN.DATASET.AUGMENTATION_TYPE = None
_CN.DATASET.TRAIN_SHUFFLE = True
_CN.DATASET.USE_OVERLAPS = True

_CN.DATASET.MATTERPORT_HORIZONTAL_IMGS_ONLY = True
_CN.DATASET.MATTERPORT_NORMALIZE = "imagenet"


############### TRAINING ###############
_CN.TRAINER = CN()
_CN.TRAINER.SEED = 42
_CN.TRAINER.GRADIENT_CLIPPING = None
_CN.TRAINER.ACCUMULATE_GRAD_BATCHES = 1

_CN.TRAIN = CN()
_CN.TRAIN.N_LANDMARKS = 64
_CN.TRAIN.FRAC_PATCH_SUBSAMPLE = 0.2
_CN.TRAIN.KEEP_ALL_POSITIVE_PATCHES = True


############### OPTIMIZER ###############
_CN.OPTIMIZER = CN()
_CN.OPTIMIZER.TYPE = "Adam"
_CN.OPTIMIZER.LR = 1e-3
_CN.OPTIMIZER.WEIGHT_DECAY = 0.0


############## LOSS ###############
_CN.LOSS = CN()
_CN.LOSS.SIGMOID_TEMPERATURE = 0.01


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _CN.clone()
