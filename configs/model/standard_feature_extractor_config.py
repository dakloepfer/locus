from yacs.config import CfgNode as CN

cfg = CN()

############### MODEL ###############
cfg.MODEL_TYPE = "dino"

cfg.MODEL = CN()

cfg.MODEL.N_PRETRAINED_BLOCKS = 11  # number of pretrained DINO blocks
cfg.MODEL.VIT_BACKBONE = "vitb8"

cfg.MODEL.FEATURE_DIM = 64
cfg.MODEL.POS_LANDMARK_RADIUS = 0.2
cfg.MODEL.NEG_LANDMARK_RADIUS = 10.0
cfg.MODEL.LANDMARK_EMBEDDING_METHOD = "sampled_patch"


############### TRAINING ###############
cfg.TRAINER = CN()
cfg.TRAINER.SEED = 42
cfg.TRAINER.GRADIENT_CLIPPING = None
cfg.TRAINER.ACCUMULATE_GRAD_BATCHES = 1

cfg.TRAIN = CN()
cfg.TRAIN.N_LANDMARKS = 64
cfg.TRAIN.FRAC_PATCH_SUBSAMPLE = 0.1
cfg.TRAIN.KEEP_ALL_POSITIVE_PATCHES = True

############### OPTIMIZER ###############
cfg.OPTIMIZER = CN()
cfg.OPTIMIZER.TYPE = "Adam"
cfg.OPTIMIZER.LR = 1e-3
cfg.OPTIMIZER.WEIGHT_DECAY = 0.0


############## LOSS ###############
cfg.LOSS = CN()
cfg.LOSS.SIGMOID_TEMPERATURE = 0.01
