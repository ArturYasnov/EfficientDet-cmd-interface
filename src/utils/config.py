class CFG:
    BASE_PATH = "."
    MODEL_SAVE_DIR = f"{BASE_PATH}/models"
    DATA_PATH = f"{BASE_PATH}/data"


class TRAIN_CFG:
    # this experiment
    log_dir = f'{CFG.BASE_PATH}/logs/'
    exp_name = 'validating map per class'
    device = "cuda"

    # train config
    train_aug = "train"
    img_size = 512

    cutmix_proba = 0

    epochs = 30
    train_bs = 16
    valid_bs = 4
    accumulate_bs = 1
    lr = 2.0

    # scheduler1
    step = [50, 90]
    step_size=3
    gamma=0.2

    scheduler = 'CosineAnnealingLR'
    t_0 = 50  # int(epochs * 1.1) + 5
    t_mul = 2
    validate_map = False

    DATA_PATH_TRAIN = "data/train"
    DATA_PATH_VALID = "data/valid"

    num_classes = 7