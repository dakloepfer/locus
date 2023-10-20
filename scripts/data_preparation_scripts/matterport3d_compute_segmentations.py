"""Use the triangle meshes and region_segmentations information to for each image give the semantic segmentation labels for each pixel."""

import os
import argparse
from tqdm import tqdm
import json
import csv

import numpy as np
from PIL import Image
import open3d

from einops import rearrange

ALL_SCENES = [
    "17DRP5sb8fy",
    "1LXtFkjw3qL",
    "1pXnuDYAj8r",
    "29hnd4uzFmX",
    "2azQ1b91cZZ",
    "2n8kARJN3HM",
    "2t7WUuJeko7",
    "5LpN3gDmAk7",
    "5q7pvUzZiYa",
    "5ZKStnWn8Zo",
    "759xd9YjKW5",
    "7y3sRwLe3Va",
    "8194nk5LbLH",
    "82sE5b5pLXE",
    "8WUmhLawc2A",
    "aayBHfsNo7d",
    "ac26ZMwG7aT",
    "ARNzJeq3xxb",
    "B6ByNegPMKs",
    "b8cTxDM8gDG",
    "cV4RVeZvu5T",
    "D7G3Y4RVNrH",
    "D7N2EKCX4Sj",
    "dhjEzFoUFzH",
    "E9uDoFAP3SH",
    "e9zR4mvMWw7",
    "EDJbREhghzL",
    "EU6Fwq7SyZv",
    "fzynW3qQPVF",
    "GdvgFV5R1Z5",
    "gTV8FGcVJC9",
    "gxdoqLR6rwA",
    "gYvKGZ5eRqb",
    "gZ6f7yhEvPG",
    "HxpKQynjfin",
    "i5noydFURQK",
    "JeFG25nYj2p",
    "JF19kD82Mey",
    "jh4fc5c5qoQ",
    "JmbYfDe2QKZ",
    "jtcxE69GiFV",
    "kEZ7cmS4wCh",
    "mJXqzFtmKg4",
    "oLBMNvg9in8",
    "p5wJjkQkbXX",
    "pa4otMbVnkk",
    "pLe4wQe7qrG",
    "Pm6F8kyY3z2",
    "pRbA3pwrgk9",
    "PuKPg4mmafe",
    "PX4nDJXEHrG",
    "q9vSo1VnCiC",
    "qoiz87JEwZ2",
    "QUCTc6BB5sX",
    "r1Q1Z4BcV1o",
    "r47D5H71a5s",
    "rPc6DW4iMge",
    "RPmz2sHmrrY",
    "rqfALeAoiTq",
    "s8pcmisQ38h",
    "S9hNv5qa7GM",
    "sKLMLpTHeUy",
    "SN83YJsR3w2",
    "sT4fr6TAbpF",
    "TbHJrupSAjP",
    "ULsKaCPVFJR",
    "uNb9QFRL6hY",
    "ur6pFq6Qu1A",
    "UwV83HsGsw3",
    "Uxmj2M2itWa",
    "V2XKFyX4ASd",
    "VFuaQ6m2Qom",
    "VLzqgDo317F",
    "Vt2qJdWjCF2",
    "VVfe2KiqLaN",
    "Vvot9Ly1tCj",
    "vyrNrziPKCB",
    "VzqfbhrpDEA",
    "wc2JMjhGNzB",
    "WYY7iVyf5p8",
    "X7HyMhZNoso",
    "x8F5xyUWy9e",
    "XcA2TqTSSAj",
    "YFuZgdQ5vWj",
    "YmJkqBEsHnH",
    "yqstnuAEVhm",
    "YVUC4YcDtcY",
    "Z6MFQCViBuw",
    "ZMojNkEp431",
    "zsNo4HB9uLZ",
]

# NOTE: had to clean up some of the labels; the only one that is maybe not completely obvious is replacing the "roof or floor / other room" with "ceiling"
MPCAT40_TO_INDEX = {
    "void": 0,
    "wall": 1,
    "floor": 2,
    "chair": 3,
    "door": 4,
    "table": 5,
    "picture": 6,
    "cabinet": 7,
    "cushion": 8,
    "window": 9,
    "sofa": 10,
    "bed": 11,
    "curtain": 12,
    "chest_of_drawers": 13,
    "plant": 14,
    "sink": 15,
    "stairs": 16,
    "ceiling": 17,
    "toilet": 18,
    "stool": 19,
    "towel": 20,
    "mirror": 21,
    "tv_monitor": 22,
    "shower": 23,
    "column": 24,
    "bathtub": 25,
    "counter": 26,
    "fireplace": 27,
    "lighting": 28,
    "beam": 29,
    "railing": 30,
    "shelving": 31,
    "blinds": 32,
    "gym_equipment": 33,
    "seating": 34,
    "board_panel": 35,
    "furniture": 36,
    "appliances": 37,
    "clothes": 38,
    "objects": 39,
    "misc": 40,
    "unlabeled": 41,
}
STUFF_CLASSES = {
    "void": 0,
    "wall": 1,
    "floor": 2,
    "curtain": 3,
    "stairs": 4,
    "ceiling": 5,
    "mirror": 6,
    "shower": 7,
    "column": 8,
    "beam": 9,
    "railing": 10,
    "shelving": 11,
    "blinds": 12,
    "board_panel": 13,
    "misc": 14,
    "unlabeled": 15,
}

STUFF_CLASSES_V1 = {
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

INVALID_ID = open3d.t.geometry.RaycastingScene.INVALID_ID  # 4294967295
INVALID_CATEGORY = 41  # the unlabeled index

CATEGORY_TO_HEX = {
    0: "#ffffff",
    1: "#aec7e8",
    2: "#708090",
    3: "#98df8a",
    4: "#c5b0d5",
    5: "#ff7f0e",
    6: "#d62728",
    7: "#1f77b4",
    8: "#bcbd22",
    9: "#ff9896",
    10: "#2ca02c",
    11: "#e377c2",
    12: "#de9ed6",
    13: "#9467bd",
    14: "#8ca252",
    15: "#843c39",
    16: "#9edae5",
    17: "#9c9ede",
    18: "#e7969c",
    19: "#637939",
    20: "#8c564b",
    21: "#dbdb8d",
    22: "#d6616b",
    23: "#cedb9c",
    24: "#e7ba52",
    25: "#393b79",
    26: "#a55194",
    27: "#ad494a",
    28: "#b5cf6b",
    29: "#5254a3",
    30: "#bd9e39",
    31: "#c49c94",
    32: "#f7b6d2",
    33: "#6b6ecf",
    34: "#ffbb78",
    35: "#c7c7c7",
    36: "#8c6d31",
    37: "#e7cb94",
    38: "#ce6dbd",
    39: "#17becf",
    40: "#7f7f7f",
    41: "#000000",
}


def hex_to_rgb(hex_string):
    """Convert a hex color string to an RGB color.

    Parameters
    ----------
    hex_string (str)
        The hex color string.

    Returns
    -------
    numpy array length 3
        The RGB color.
    """
    hex_string = hex_string.lstrip("#")
    return np.array(
        [int(hex_string[i : i + 2], base=16) for i in (0, 2, 4)], dtype=np.uint8
    )


def visualise_segmentation(segmentation_map, save_path):
    image = np.zeros(
        (segmentation_map.shape[0], segmentation_map.shape[1], 3), dtype=np.uint8
    )
    for i in range(segmentation_map.shape[0]):
        for j in range(segmentation_map.shape[1]):
            image[i, j, :] = hex_to_rgb(
                CATEGORY_TO_HEX[segmentation_map[i, j] % len(CATEGORY_TO_HEX)]
            )

    # save image
    img = Image.fromarray(image)
    img.save(save_path)


def invert_se3(se3_matrix):
    """Invert an SE(3) transformations.

    Parameters
    ----------
    se3_matrix (4 x 4 tensor)
        The SE(3) transformations to invert.

    Returns
    -------
    4 x 4 tensor
        The inverted transformation.
    """

    R = se3_matrix[:3, :3]
    t = se3_matrix[:3, 3:]

    inverse = np.eye(4, dtype=se3_matrix.dtype)

    inverse_R = R.T
    inverse[:3, :3] = inverse_R
    inverse[:3, 3:] = -inverse_R @ t

    return inverse


def calc_pixel_triangle_mapping(raycaster, camera_pose, intrinsic_matrix, opts):
    # the camera pose is the camera-to-world transformation, so invert_se3(camera_pose) is the world-to-camera transformation.
    # But Matterport3d uses a different camera-coordinate system than Open3D (Matterport uses y up, z into camera, x right, Open3D uses y down, z out of camera, x right)
    # so to get the world-to-Open3D camera transformation, we need to flip the y and z axes.
    extrinsic_matrix = np.array(
        [[1.0, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    ) @ invert_se3(camera_pose)
    extrinsic_matrix = open3d.core.Tensor(extrinsic_matrix)

    # also, the principal point for y is now measured from the top of the image, not the bottom
    intrinsic_matrix[1, 2] = opts.img_height - intrinsic_matrix[1, 2]
    intrinsic_matrix = open3d.core.Tensor(intrinsic_matrix)

    rays = open3d.t.geometry.RaycastingScene.create_rays_pinhole(
        intrinsic_matrix, extrinsic_matrix, opts.img_width, opts.img_height
    )

    return raycaster.cast_rays(rays)["primitive_ids"].numpy()


def load_fsegs_and_segs_to_cat(
    opts, all_regions, scene_dir, raw_semantic_label_to_mpcat40
):
    """Creates maps from triangle indices to segment indices and segment indices to category indices.

    Returns:
    --------
    triangle_to_segment ((n_triangles, ) numpy array):
        the segment index for each triangle

    segment_to_category ((n_segments, ) numpy array):
        the category index for each segment
    """

    # load triangle -> segment mappings
    # this should be a numpy array of shape (num_triangles,), where each entry is the segment index of the triangle

    n_regions = len(all_regions)
    all_fsegs = []
    all_seggroups = []
    for region in all_regions:
        with open(
            os.path.join(scene_dir, "region_segmentations", region + ".fsegs.json")
        ) as f:
            all_fsegs.append(json.load(f)["segIndices"])

        with open(
            os.path.join(scene_dir, "region_segmentations", region + ".semseg.json")
        ) as f:
            all_seggroups.append(json.load(f)["segGroups"])

    # shift the segment indices so that they are unique across meshes
    for i in range(n_regions):
        all_fsegs[i] = np.array(all_fsegs[i], dtype=np.int32)

        if i > 0:
            index_shift = np.max(all_fsegs[i - 1]) + 1
            all_fsegs[i] += index_shift

            for j in range(len(all_seggroups[i])):
                # go through all object in the region
                all_seggroups[i][j]["segments"] = (
                    np.array(all_seggroups[i][j]["segments"]) + index_shift
                )

    # add a new segment for invalid ids -- basically all rays that don't hit a triangle hit this appended phantom triangle with a phantom segment
    # this might end up not being needed if they use the "void" category for all this, but I can't be sure of that
    all_fsegs.append(np.array([all_fsegs[-1].max() + 1]))
    triangle_to_segment = np.concatenate(all_fsegs, axis=0)

    n_segments = triangle_to_segment.max() + 1

    # a bunch of segments have no object (don't appear in the segGroups file), so start out with putting them all as INVALID_CATEGORY
    if opts.instance_segmentation:
        segment_to_category = np.ones(n_segments, dtype=np.int32) * (-1)
    else:
        segment_to_category = np.ones(n_segments, dtype=np.int32) * INVALID_CATEGORY

    for i in range(n_regions):
        if opts.instance_segmentation and i > 0:
            index_shift = np.max(segment_to_category) + 1
        else:
            index_shift = len(STUFF_CLASSES)

        n_stuff_objects = 0
        for j in range(len(all_seggroups[i])):
            if opts.instance_segmentation:
                # different categories for individual instances of objects of the same category
                # add the max category index to the object id to get a unique category index for each instance even across regions

                if (
                    raw_semantic_label_to_mpcat40[all_seggroups[i][j]["label"]]
                    in STUFF_CLASSES.keys()
                ):  # stuff objects are all mapped to their semantic class
                    category_idx = STUFF_CLASSES[
                        raw_semantic_label_to_mpcat40[all_seggroups[i][j]["label"]]
                    ]
                    n_stuff_objects += 1
                else:  # not stuff, so every object is its own category
                    category_idx = (
                        all_seggroups[i][j]["objectId"] + index_shift - n_stuff_objects
                    )

            else:
                category_idx = MPCAT40_TO_INDEX[
                    raw_semantic_label_to_mpcat40[all_seggroups[i][j]["label"]]
                ]
            segment_to_category[all_seggroups[i][j]["segments"]] = category_idx

    if opts.instance_segmentation:
        # change the category for invalid objects (the ones that are mapped to -1) to a positive number, ie the max category index + 1
        segment_to_category[segment_to_category < 0] = np.max(segment_to_category) + 1

    return triangle_to_segment, segment_to_category


def load_meshes(all_regions, scene_dir):
    # load the meshes
    all_meshes = []
    for region in all_regions:
        mesh = open3d.io.read_triangle_mesh(
            os.path.join(scene_dir, "region_segmentations", region + ".ply")
        )
        all_meshes.append(mesh)

    # combine meshes, taking care of segment and category mappings
    # (create RayMeshIntersector here)
    scene_mesh = all_meshes[0]
    for mesh in all_meshes[1:]:
        scene_mesh += mesh

    return scene_mesh


def main(opts):
    raw_semantic_label_to_mpcat40 = {}

    with open(
        os.path.join(opts.category_mapping_dir, "category_mapping.tsv"), "r"
    ) as tsvfile:
        reader = csv.reader(tsvfile, delimiter="\t")

        # Skip header row
        next(reader)
        for row in reader:
            # Map second column to last column
            raw_semantic_label_to_mpcat40[row[1]] = row[-1]

    for scene in ALL_SCENES:
        print(f"Processing scene {scene}")
        scene_dir = os.path.join(opts.raw_dataset_dir, scene)

        if opts.instance_segmentation:
            if not os.path.exists(os.path.join(scene_dir, "object_maps")):
                os.makedirs(os.path.join(scene_dir, "object_maps"))

        else:
            if not os.path.exists(os.path.join(scene_dir, "semantic_maps")):
                os.makedirs(os.path.join(scene_dir, "semantic_maps"))

        all_files = os.listdir(os.path.join(scene_dir, "region_segmentations"))
        all_regions = list(
            set([f.split(".")[0] for f in all_files if f.startswith("region")])
        )
        all_regions.sort()

        triangle_to_segment, segment_to_category = load_fsegs_and_segs_to_cat(
            opts, all_regions, scene_dir, raw_semantic_label_to_mpcat40
        )
        scene_mesh = load_meshes(all_regions, scene_dir)

        # For some reason open3d has more than one TriangleMesh class
        raycaster_mesh = open3d.t.geometry.TriangleMesh(
            open3d.core.Tensor(np.asarray(scene_mesh.vertices).astype(np.float32)),
            open3d.core.Tensor(np.asarray(scene_mesh.triangles).astype(np.float32)),
        )

        raycaster = open3d.t.geometry.RaycastingScene()
        raycaster.add_triangles(raycaster_mesh)

        # check that triangle_to_segment has length num_triangles + 1 (for the phantom triangle)
        assert (
            triangle_to_segment.shape[0]
            == np.asarray(scene_mesh.triangles).shape[0] + 1
        )
        # check that segment_to_category has length num_segments (including the phantom segment, which is already included in triangle_to_segment)
        assert segment_to_category.shape[0] == np.max(triangle_to_segment) + 1

        # load camera poses
        camera_parameter_file_path = os.path.join(
            scene_dir, "undistorted_camera_parameters", "{}.conf".format(scene)
        )

        with open(camera_parameter_file_path, "r") as f:
            params_file_lines = f.readlines()

        current_intrinsics = None
        image_names = []
        camera_poses = []
        intrinsic_matrices = []

        all_used_img_files = os.listdir(
            os.path.join(scene_dir, "rgb_highres" if opts.img_height == 448 else "rgb")
        )

        for line in params_file_lines:
            if line.startswith("intrinsics"):
                intrinsics_line = [
                    i
                    for i in line.strip().split(" ")
                    if not (i.isspace() or len(i) == 0)
                ]

                current_intrinsics = np.array([float(i) for i in intrinsics_line[1:]])
                current_intrinsics = rearrange(
                    current_intrinsics, "(h w) -> h w", h=3, w=3
                )
                # adjust for larger pixels
                current_intrinsics[0] *= opts.img_width / 1280
                current_intrinsics[1] *= opts.img_height / 1024

            elif line.startswith("scan"):
                scan_line = line.split(" ")[1:]

                img_file_name = scan_line[1].replace(".jpg", ".npy")

                # I filter out some images that had too few good depth values to inpaint the depth map
                if img_file_name not in all_used_img_files:
                    continue

                cam_pose = np.array([float(i) for i in scan_line[2:]])
                cam_pose = rearrange(cam_pose, "(h w) -> h w", h=4, w=4)

                image_names.append(img_file_name[:-4])
                intrinsic_matrices.append(current_intrinsics)
                camera_poses.append(cam_pose)

            else:
                continue

        # do this for each image, as I need to save each image's category indices separately
        for i in tqdm(
            range(len(camera_poses)),
            desc="Calculating segmentation maps for scene {}".format(scene),
        ):
            camera_pose = camera_poses[i]
            intrinsic_matrix = intrinsic_matrices[i]
            image_name = image_names[i]

            triangle_indices = calc_pixel_triangle_mapping(
                raycaster, camera_pose, intrinsic_matrix, opts
            )
            triangle_indices[triangle_indices == INVALID_ID] = triangle_to_segment.max()

            segment_indices = triangle_to_segment[triangle_indices]
            category_indices = segment_to_category[segment_indices]

            if opts.debug:
                if not os.path.exists("./test_segm/{}".format(scene)):
                    os.makedirs("./test_segm/{}".format(scene))
                visualise_segmentation(
                    category_indices,
                    "./test_segm/{}/".format(scene) + image_name + ".png",
                )

                command = input(
                    'Press "Enter" to continue to next image, "ns" to go to the next scene...'
                )
                if command == "ns":
                    break
            else:
                if opts.instance_segmentation:
                    np.save(
                        os.path.join(scene_dir, "object_maps", image_name),
                        category_indices,
                    )
                else:
                    np.save(
                        os.path.join(scene_dir, "semantic_maps", image_name),
                        category_indices,
                    )

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add("-d", "--debug", action="store_true", help="debug mode", default=False)
    parser.add(
        "-i",
        "--instance_segmentation",
        action="store_true",
        help="segment individual instances of objects, not categories",
        default=True,
    )
    parser.add_argument(
        "--raw_dataset_dir",
        type=str,
        default="/data/matterport3d",
    )
    parser.add_argument(
        "--category_mapping_dir",
        type=str,
        default="./scripts/data_preparation_scripts",
    )
    parser.add_argument("--img_width", type=int, default=320)
    parser.add_argument("--img_height", type=int, default=256)
    opts = parser.parse_args()
    main(opts)
