import albumentations as A


class DarkAug(object):
    """
    Extreme dark augmentation aiming at Aachen Day-Night
    """

    def __init__(self) -> None:
        self.augmentor = A.Compose(
            [
                A.RandomBrightnessContrast(
                    p=0.75, brightness_limit=(-0.6, 0.0), contrast_limit=(-0.5, 0.3)
                ),
                A.Blur(p=0.1, blur_limit=(3, 9)),
                A.MotionBlur(p=0.2, blur_limit=(3, 25)),
                A.RandomGamma(p=0.1, gamma_limit=(15, 65)),
                A.HueSaturationValue(p=0.1, val_shift_limit=(-100, -40)),
            ],
            p=0.75,
        )

    def __call__(self, x):
        return self.augmentor(image=x)["image"]


class MobileAug(object):
    """
    Random augmentations aiming at images of mobile/handhold devices.
    """

    def __init__(self, outdoor=True):
        if outdoor:
            self.augmentor = A.Compose(
                [
                    A.MotionBlur(p=0.15),
                    A.ColorJitter(p=0.5),
                    A.RandomRain(p=0.05),  # random occlusion
                    A.RandomSunFlare(p=0.1),
                    A.JpegCompression(p=0.25),
                    A.ISONoise(p=0.25),
                ],
                p=1.0,
            )
        else:
            self.augmentor = A.Compose(
                [
                    A.MotionBlur(p=0.15),
                    A.ColorJitter(p=0.5),
                    A.JpegCompression(p=0.25),
                    A.ISONoise(p=0.25),
                ],
                p=1.0,
            )

    def __call__(self, x):
        return self.augmentor(image=x)["image"]


class DroneAug(object):
    """
    Random augmentations aiming at images taken by drones.
    """

    def __init__(self, outdoor=True):
        if outdoor:
            self.augmentor = A.Compose(
                [
                    A.MotionBlur(p=0.25),
                    A.ColorJitter(p=0.5),
                    A.RandomRain(p=0.1),  # random occlusion
                    A.RandomSunFlare(p=0.1),
                    A.JpegCompression(p=0.25),
                    A.ISONoise(p=0.25),
                ],
                p=1.0,
            )
        else:
            self.augmentor = A.Compose(
                [
                    A.MotionBlur(p=0.25),
                    A.ColorJitter(p=0.5),
                    A.JpegCompression(p=0.25),
                    A.ISONoise(p=0.25),
                ],
                p=1.0,
            )

    def __call__(self, x):
        return self.augmentor(image=x)["image"]


def build_augmentor(method=None, data_source=None, **kwargs):
    if method == "dark":
        return DarkAug()
    elif method == "mobile":
        if data_source.lower() in ["eurocmav", "scannet", "inloc"]:
            outdoor = False
        else:
            outdoor = True

        return MobileAug(outdoor=outdoor)
    elif method == "drone":
        if data_source.lower() in ["eurocmav", "scannet", "inloc"]:
            outdoor = False
        else:
            outdoor = True

        return DroneAug(outdoor=outdoor)

    elif method is None:
        return None
    else:
        raise ValueError(f"Invalid augmentation method: {method}")


if __name__ == "__main__":
    augmentor = build_augmentor("FDA")
