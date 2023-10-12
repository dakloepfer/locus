import os

import numpy as np

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_tensor

# number of objects, with all "stuff" classes grouped into one object per class
N_OBJECTS_PER_SCENE = {
    "17DRP5sb8fy": 120,
    "1LXtFkjw3qL": 279,
    "1pXnuDYAj8r": 588,
    "29hnd4uzFmX": 119,
    "2azQ1b91cZZ": 522,
    "2n8kARJN3HM": 512,
    "2t7WUuJeko7": 121,
    "5LpN3gDmAk7": 273,
    "5q7pvUzZiYa": 237,
    "5ZKStnWn8Zo": 598,
    "759xd9YjKW5": 432,
    "7y3sRwLe3Va": 418,
    "8194nk5LbLH": 109,
    "82sE5b5pLXE": 297,
    "8WUmhLawc2A": 356,
    "aayBHfsNo7d": 298,
    "ac26ZMwG7aT": 643,
    "ARNzJeq3xxb": 265,
    "B6ByNegPMKs": 1194,
    "b8cTxDM8gDG": 436,
    "cV4RVeZvu5T": 209,
    "D7G3Y4RVNrH": 82,
    "D7N2EKCX4Sj": 736,
    "dhjEzFoUFzH": 60,
    "E9uDoFAP3SH": 622,
    "e9zR4mvMWw7": 303,
    "EDJbREhghzL": 313,
    "EU6Fwq7SyZv": 247,
    "fzynW3qQPVF": 221,
    "GdvgFV5R1Z5": 66,
    "gTV8FGcVJC9": 710,
    "gxdoqLR6rwA": 37,
    "gYvKGZ5eRqb": 49,
    "gZ6f7yhEvPG": 24,
    "HxpKQynjfin": 79,
    "i5noydFURQK": 209,
    "JeFG25nYj2p": 245,
    "JF19kD82Mey": 134,
    "jh4fc5c5qoQ": 155,
    "JmbYfDe2QKZ": 329,
    "jtcxE69GiFV": 578,
    "kEZ7cmS4wCh": 344,
    "mJXqzFtmKg4": 381,
    "oLBMNvg9in8": 390,
    "p5wJjkQkbXX": 406,
    "pa4otMbVnkk": 773,
    "pLe4wQe7qrG": 22,
    "Pm6F8kyY3z2": 61,
    "pRbA3pwrgk9": 260,
    "PuKPg4mmafe": 271,
    "PX4nDJXEHrG": 604,
    "q9vSo1VnCiC": 262,
    "qoiz87JEwZ2": 332,
    "QUCTc6BB5sX": 527,
    "r1Q1Z4BcV1o": 345,
    "r47D5H71a5s": 139,
    "rPc6DW4iMge": 306,
    "RPmz2sHmrrY": 150,
    "rqfALeAoiTq": 188,
    "s8pcmisQ38h": 143,
    "S9hNv5qa7GM": 288,
    "sKLMLpTHeUy": 241,
    "SN83YJsR3w2": 666,
    "sT4fr6TAbpF": 206,
    "TbHJrupSAjP": 380,
    "ULsKaCPVFJR": 265,
    "uNb9QFRL6hY": 492,
    "ur6pFq6Qu1A": 298,
    "UwV83HsGsw3": 236,
    "Uxmj2M2itWa": 167,
    "V2XKFyX4ASd": 254,
    "VFuaQ6m2Qom": 440,
    "VLzqgDo317F": 387,
    "Vt2qJdWjCF2": 206,
    "VVfe2KiqLaN": 151,
    "Vvot9Ly1tCj": 362,
    "vyrNrziPKCB": 846,
    "VzqfbhrpDEA": 810,
    "wc2JMjhGNzB": 582,
    "WYY7iVyf5p8": 226,
    "X7HyMhZNoso": 246,
    "x8F5xyUWy9e": 124,
    "XcA2TqTSSAj": 235,
    "YFuZgdQ5vWj": 257,
    "YmJkqBEsHnH": 67,
    "yqstnuAEVhm": 527,
    "YVUC4YcDtcY": 156,
    "Z6MFQCViBuw": 351,
    "ZMojNkEp431": 219,
    "zsNo4HB9uLZ": 279,
}

STUFF_CLASSES = {
    "void": 0,
    "wall": 1,
    "floor": 2,
    "ceiling": 3,
    "column": 4,
    "beam": 5,
    "objects": 6,
    "misc": 7,
    "unlabeled": 8,
}

N_STUFF_CLASS_OBJECTS = {
    "17DRP5sb8fy": [15, 10, 12, 10, 0, 0, 14, 4, 10],
    "1LXtFkjw3qL": [10, 104, 31, 33, 6, 6, 54, 4, 31],
    "1pXnuDYAj8r": [18, 125, 36, 34, 10, 3, 108, 20, 29],
    "29hnd4uzFmX": [3, 54, 15, 8, 0, 7, 34, 10, 14],
    "2azQ1b91cZZ": [18, 126, 35, 37, 2, 4, 77, 22, 28],
    "2n8kARJN3HM": [15, 56, 34, 32, 0, 1, 72, 40, 28],
    "2t7WUuJeko7": [2, 6, 6, 6, 0, 0, 36, 5, 6],
    "5LpN3gDmAk7": [19, 62, 33, 26, 4, 0, 30, 2, 28],
    "5q7pvUzZiYa": [16, 47, 22, 19, 1, 0, 51, 7, 19],
    "5ZKStnWn8Zo": [28, 142, 32, 33, 5, 0, 142, 21, 28],
    "759xd9YjKW5": [2, 92, 29, 28, 0, 12, 45, 14, 19],
    "7y3sRwLe3Va": [13, 156, 40, 50, 0, 6, 61, 31, 35],
    "8194nk5LbLH": [1, 35, 9, 18, 3, 2, 17, 6, 8],
    "82sE5b5pLXE": [3, 86, 20, 21, 7, 0, 91, 9, 16],
    "8WUmhLawc2A": [6, 94, 27, 27, 0, 4, 86, 9, 18],
    "aayBHfsNo7d": [12, 86, 28, 21, 1, 0, 55, 14, 21],
    "ac26ZMwG7aT": [10, 193, 40, 37, 0, 4, 295, 47, 31],
    "ARNzJeq3xxb": [23, 68, 23, 18, 5, 1, 69, 19, 19],
    "B6ByNegPMKs": [9, 341, 83, 241, 65, 1, 180, 77, 81],
    "b8cTxDM8gDG": [13, 116, 38, 35, 1, 1, 69, 7, 24],
    "cV4RVeZvu5T": [15, 92, 16, 18, 0, 0, 39, 21, 11],
    "D7G3Y4RVNrH": [4, 32, 6, 6, 0, 0, 5, 9, 5],
    "D7N2EKCX4Sj": [37, 232, 43, 63, 26, 12, 328, 50, 47],
    "dhjEzFoUFzH": [0, 24, 9, 3, 0, 0, 2, 4, 4],
    "E9uDoFAP3SH": [20, 226, 46, 40, 3, 0, 56, 17, 40],
    "e9zR4mvMWw7": [25, 56, 24, 26, 1, 0, 29, 4, 24],
    "EDJbREhghzL": [15, 78, 15, 15, 1, 0, 73, 22, 15],
    "EU6Fwq7SyZv": [7, 50, 21, 11, 7, 8, 65, 14, 17],
    "fzynW3qQPVF": [9, 129, 33, 31, 0, 0, 25, 7, 31],
    "GdvgFV5R1Z5": [8, 9, 6, 6, 0, 0, 4, 3, 6],
    "gTV8FGcVJC9": [18, 191, 52, 65, 0, 2, 111, 34, 51],
    "gxdoqLR6rwA": [0, 29, 13, 3, 24, 0, 12, 3, 4],
    "gYvKGZ5eRqb": [0, 9, 2, 0, 14, 0, 21, 0, 1],
    "gZ6f7yhEvPG": [0, 5, 2, 1, 0, 0, 6, 0, 1],
    "HxpKQynjfin": [11, 16, 10, 11, 0, 0, 10, 1, 8],
    "i5noydFURQK": [12, 33, 14, 15, 0, 0, 23, 4, 14],
    "JeFG25nYj2p": [21, 68, 26, 23, 0, 0, 41, 4, 25],
    "JF19kD82Mey": [10, 61, 17, 13, 0, 0, 32, 0, 15],
    "jh4fc5c5qoQ": [4, 28, 20, 16, 1, 0, 37, 1, 16],
    "JmbYfDe2QKZ": [10, 60, 23, 26, 3, 0, 46, 8, 22],
    "jtcxE69GiFV": [26, 122, 37, 53, 0, 1, 54, 29, 44],
    "kEZ7cmS4wCh": [48, 94, 37, 35, 2, 1, 42, 14, 41],
    "mJXqzFtmKg4": [32, 61, 23, 30, 11, 0, 63, 25, 25],
    "oLBMNvg9in8": [24, 88, 48, 45, 4, 1, 91, 57, 31],
    "p5wJjkQkbXX": [41, 84, 39, 50, 1, 0, 46, 31, 38],
    "pa4otMbVnkk": [52, 181, 46, 66, 7, 2, 112, 35, 51],
    "pLe4wQe7qrG": [0, 6, 2, 3, 6, 0, 2, 0, 2],
    "Pm6F8kyY3z2": [0, 9, 5, 6, 4, 0, 3, 1, 4],
    "pRbA3pwrgk9": [2, 88, 31, 42, 0, 0, 50, 13, 28],
    "PuKPg4mmafe": [0, 27, 7, 8, 4, 0, 11, 0, 7],
    "PX4nDJXEHrG": [4, 224, 54, 57, 3, 0, 74, 8, 50],
    "q9vSo1VnCiC": [0, 105, 25, 25, 0, 2, 29, 6, 21],
    "qoiz87JEwZ2": [2, 66, 23, 21, 16, 0, 53, 10, 21],
    "QUCTc6BB5sX": [1, 94, 40, 37, 0, 0, 45, 18, 36],
    "r1Q1Z4BcV1o": [3, 62, 12, 16, 11, 5, 11, 8, 14],
    "r47D5H71a5s": [1, 62, 22, 21, 0, 0, 28, 4, 21],
    "rPc6DW4iMge": [8, 73, 33, 33, 4, 0, 23, 8, 27],
    "RPmz2sHmrrY": [5, 26, 8, 9, 0, 0, 48, 6, 8],
    "rqfALeAoiTq": [3, 98, 31, 30, 3, 0, 23, 17, 26],
    "s8pcmisQ38h": [2, 53, 18, 19, 0, 0, 15, 2, 17],
    "S9hNv5qa7GM": [9, 76, 21, 24, 0, 0, 16, 9, 19],
    "sKLMLpTHeUy": [6, 102, 27, 30, 0, 0, 28, 9, 27],
    "SN83YJsR3w2": [12, 226, 56, 68, 15, 2, 77, 22, 52],
    "sT4fr6TAbpF": [11, 43, 22, 26, 1, 1, 21, 10, 20],
    "TbHJrupSAjP": [28, 90, 34, 40, 0, 1, 34, 15, 28],
    "ULsKaCPVFJR": [6, 54, 26, 28, 0, 0, 55, 8, 25],
    "uNb9QFRL6hY": [13, 139, 54, 64, 8, 13, 65, 10, 50],
    "ur6pFq6Qu1A": [4, 98, 34, 34, 1, 0, 63, 15, 30],
    "UwV83HsGsw3": [21, 78, 30, 33, 0, 0, 37, 7, 27],
    "Uxmj2M2itWa": [3, 48, 27, 28, 1, 0, 18, 9, 27],
    "V2XKFyX4ASd": [9, 84, 33, 26, 8, 6, 17, 16, 29],
    "VFuaQ6m2Qom": [4, 106, 39, 35, 0, 0, 39, 12, 31],
    "VLzqgDo317F": [5, 95, 21, 21, 10, 2, 50, 34, 19],
    "Vt2qJdWjCF2": [3, 105, 29, 28, 10, 0, 93, 28, 41],
    "VVfe2KiqLaN": [7, 44, 12, 12, 19, 8, 30, 20, 12],
    "Vvot9Ly1tCj": [30, 137, 31, 46, 3, 0, 55, 29, 29],
    "vyrNrziPKCB": [111, 272, 71, 70, 28, 2, 230, 108, 84],
    "VzqfbhrpDEA": [101, 320, 59, 78, 0, 1, 247, 65, 63],
    "wc2JMjhGNzB": [62, 159, 46, 58, 6, 0, 76, 54, 42],
    "WYY7iVyf5p8": [8, 72, 31, 33, 0, 0, 10, 6, 23],
    "X7HyMhZNoso": [15, 153, 31, 27, 0, 0, 28, 24, 21],
    "x8F5xyUWy9e": [7, 62, 14, 14, 3, 0, 35, 13, 13],
    "XcA2TqTSSAj": [5, 110, 36, 37, 0, 2, 24, 4, 33],
    "YFuZgdQ5vWj": [15, 57, 20, 21, 2, 0, 32, 7, 16],
    "YmJkqBEsHnH": [1, 8, 2, 3, 12, 3, 9, 6, 6],
    "yqstnuAEVhm": [17, 107, 36, 42, 1, 0, 208, 39, 53],
    "YVUC4YcDtcY": [9, 28, 10, 17, 0, 0, 23, 46, 27],
    "Z6MFQCViBuw": [8, 32, 21, 20, 0, 0, 79, 26, 34],
    "ZMojNkEp431": [13, 44, 17, 28, 0, 0, 63, 26, 36],
    "zsNo4HB9uLZ": [24, 29, 22, 21, 0, 1, 83, 6, 22],
}


class MatterportSegmentationDataset(torch.utils.data.Dataset):
    """Dataset for a single Matterport3D scene, returning the image and the segmentation mask."""

    def __init__(self, data_root, scene_name, augment_fn=None, **kwargs):
        super().__init__()
        self.data_root = data_root
        self.scene_name = scene_name

        self.augment_fn = augment_fn

        self.horizontal_only = kwargs.get("horizontal_only", False)

        if kwargs.get("normalize", "imagenet") == "imagenet":
            self.normalize = transforms.Normalize(
                (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
            )  # for backbones pretrained on ImageNet
        else:
            raise NotImplementedError(
                f"Image normalization {kwargs['normalize']} not implemented."
            )

        self.img_height = kwargs.get("img_height", 1024)
        self.img_width = kwargs.get("img_width", 1024)
        self.downsampling_factor = self.img_height / 1024

        self.img_dir = None
        self.depth_dir = None

        self.data_dict = self._create_data_dict()

    def filter_img_files(self, img_dir):
        """Filter out images that aren't used for various reasons."""
        all_img_files = os.listdir(img_dir)
        used_img_files = []
        used_file_mask = torch.ones(len(all_img_files), dtype=torch.bool)

        for i, file_name in enumerate(all_img_files):
            if self.horizontal_only:
                # remove all the files that aren't looking horizontally
                if not file_name[-7] == "1":
                    used_file_mask[i] = False
                    continue

            used_img_files.append(file_name)

        return used_img_files, used_file_mask

    def _create_data_dict(self):
        data_dict = {}
        scene_dir = os.path.join(self.data_root, self.scene_name)

        if self.img_height >= 448:
            img_folder = os.path.join(scene_dir, "rgb_highres")
        else:
            img_folder = os.path.join(scene_dir, "rgb")
        segmentation_folder = os.path.join(
            scene_dir,
            "object_maps",
        )
        file_names, _ = self.filter_img_files(img_folder)

        sample_idx = 0
        for file_name in file_names:
            img_path = os.path.join(img_folder, file_name)
            segmentation_path = os.path.join(segmentation_folder, file_name)
            if not os.path.exists(segmentation_path):
                continue
            data_dict[sample_idx] = {
                "img_path": img_path,
                "segmentation_path": segmentation_path,
            }
            sample_idx += 1

        return data_dict

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, index):
        img_path = self.data_dict[index]["img_path"]
        segmentation_path = self.data_dict[index]["segmentation_path"]

        img = np.load(img_path)  # already RGB
        img = F.interpolate(
            to_tensor(img).unsqueeze(0),
            size=(self.img_height, self.img_width),
        ).squeeze(0)
        img = self.normalize(img)

        # should be a tensor that contains the segmentation categories as ints
        segmentation = torch.from_numpy(np.load(segmentation_path)).long()
        segmentation = (
            F.interpolate(
                segmentation[None, None].float(),
                size=(self.img_height, self.img_width),
                mode="nearest-exact",
            )
            .squeeze()
            .long()
        )
        if self.augment_fn is not None:
            img = self.transforms(img)

        sample = {"img": img, "segmentation": segmentation}
        return sample
