"""A script to generate better depth maps than by inpainting; this first uses the 3D meshes to fill in missing values, and only then uses inpainting to fill in the rest."""

import os
import configargparse
import zipfile
from tqdm import tqdm

import open3d
import numpy as np
from scipy import sparse
import cv2
from PIL import Image
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


def bad_depth_values(depth, min_depth=0.01, max_depth=20):
    return (depth < min_depth) | (depth > max_depth) | np.isnan(depth) | np.isinf(depth)


def inpaint_depth(depth, image, nyu_depth_dataset=False, alpha=1):
    """In-paint depth values using method from "Colorization Using Optimization" by Levin et al. (https://www.cs.huji.ac.il/w~yweiss/Colorization/), which was also used / adapted for depth inpainting in the NYU Depth Dataset (https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html).

    Parameters
    ----------
    depth (height x width numpy array):
        the depth values with some missing / noisy values

    image (height x width x channels numpy array):
        the RGB image, whose intensities are used as a guide for the inpainting

    nyu_depth_dataset (bool, optional):
        Whether I should use the method from the NYU Depth Dataset code or the one from the original paper, by default False

    alpha (int, optional):
        Only used if nyu_depth_dataset=True, this is "a penalty value betwwen 0 and 1 for the current depth values", by default 1

    Returns
    -------
    height x width numpy array:
        the inpainted depth map
    """
    # NOTE: the Matlab code that I based this on used sparse matrices, but this should be equivalent and not too much slower

    window_radius = 1
    window_size = (2 * window_radius + 1) ** 2

    height, width = depth.shape
    n_pixels = height * width

    grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).squeeze()

    depth_is_noise = bad_depth_values(depth)

    if depth_is_noise.sum() == 0:
        return depth

    if depth_is_noise.sum() >= 0.25 * n_pixels:
        raise ValueError("Not enough valid depth values to inpaint")

    # normalise depth
    max_depth = np.max(depth[~depth_is_noise])
    depth = depth / max_depth
    depth[depth > 1] = 1

    col_inds = np.zeros((n_pixels * window_size,))
    row_inds = np.zeros((n_pixels * window_size,))
    current_window_weight_vals = np.zeros((window_size,))
    vals = np.zeros(n_pixels * window_size)  # the values for the (1 - weights) matrix

    # NOTE if this takes too long (which I doubt), I could try to vectorise it; that would involve a bit more trickery with the outer_window_mask though to account for the fact that the window is not always the same size (at the edges of the image)
    # for now I just copy the for-loops from the Matlab code
    pixel_index = 0
    all_pixel_indices = np.arange(n_pixels).reshape(height, width)
    n_nonzero_vals = 0
    for i in range(height):
        for j in range(width):
            if ~depth_is_noise[i, j]:
                row_inds[n_nonzero_vals] = pixel_index
                col_inds[n_nonzero_vals] = all_pixel_indices[i, j]
                if nyu_depth_dataset:
                    vals[n_nonzero_vals] = alpha
                else:
                    vals[n_nonzero_vals] = 1

                n_nonzero_vals += 1
                pixel_index += 1
                continue

            current_window_index = 0
            for ii in range(
                max(0, i - window_radius), min(height, i + window_radius + 1)
            ):
                for jj in range(
                    max(0, j - window_radius), min(width, j + window_radius + 1)
                ):
                    if ii == i and jj == j:
                        continue

                    row_inds[n_nonzero_vals] = pixel_index
                    col_inds[n_nonzero_vals] = all_pixel_indices[ii, jj]
                    current_window_weight_vals[current_window_index] = grayscale[ii, jj]

                    n_nonzero_vals += 1
                    current_window_index += 1

            pixel_gs_value = grayscale[i, j]

            # keep that around to include it in the variance calculation
            current_window_weight_vals[current_window_index] = pixel_gs_value

            window_gs_variance = np.var(current_window_weight_vals)

            csig = 0.6 * window_gs_variance
            mgv = np.min(
                (current_window_weight_vals[:current_window_index] - pixel_gs_value)
                ** 2
            )

            if csig < (-mgv / np.log(0.01)):
                csig = -mgv / np.log(0.01)
            if csig < 2.2e-6:
                csig = 2.2e-6

            # this is not quite the formula that they use in the paper, but it's the one they use in their code
            current_window_weight_vals[:current_window_index] = np.exp(
                -(
                    (current_window_weight_vals[:current_window_index] - pixel_gs_value)
                    ** 2
                )
                / csig
            )
            current_window_weight_vals[:current_window_index] /= np.sum(
                current_window_weight_vals[:current_window_index]
            )

            # the -weights part in the (1-weights) matrix
            vals[
                n_nonzero_vals - current_window_index : n_nonzero_vals
            ] = -current_window_weight_vals[:current_window_index]

            # add the identity part of the (1-weights) matrix
            row_inds[n_nonzero_vals] = pixel_index
            col_inds[n_nonzero_vals] = all_pixel_indices[i, j]
            vals[n_nonzero_vals] = 1
            n_nonzero_vals += 1

            pixel_index += 1

    # NOTE: I would have thought that the equation should be slightly different, namely if the weights matrix is filled properly for all, then for the depth_is_noise values (1 - weights.T) @ (1 - weights) @ new_depths = 0, (with for the non-noise values still new_depths = old depths)
    # Here I solve (1 - weights) @ new_depths = 0 (with for the non-noise values still new_depths = old depths), any solution of which also solves the above equation, but maybe there is no solution to this equation but to the correct one. If I run into any errors to that effect, I might want to change this. Since this is a linear system of equations which seem pretty well-behaved, I don't expect something like this though.

    # for the non-noise pixels, if nyu_depth_dataset=False, only the diagonal is non-zero
    A = sparse.coo_matrix(
        (vals[:n_nonzero_vals], (row_inds[:n_nonzero_vals], col_inds[:n_nonzero_vals])),
        shape=(n_pixels, n_pixels),
    )
    b = np.zeros(n_pixels)
    b[(~depth_is_noise).flatten()] = depth[~depth_is_noise].flatten()

    if nyu_depth_dataset:
        # This way of handling alpha is not what the code provided by the NYU Depths authors is doing, but what they are doing seems very weird; I think their code is missing a line to skip non-noise pixels in the loop above
        # If I am correct, then I think alpha=1 is simply the original code, while alpha=0 allows the known depths to be ignored completely
        b *= alpha

    new_depths = sparse.linalg.spsolve(A.tocsc(), b)

    return rearrange(new_depths * max_depth, "(h w) -> h w", h=height, w=width)


def calc_depths_from_raycaster(raycaster, camera_pose, intrinsic_matrix, opts):
    # the camera pose is the camera-to-world transformation, so invert_se3(camera_pose) is the world-to-camera transformation.
    # But Matterport3d uses a different camera-coordinate system than Open3D (Matterport uses y up, z into camera, x right, Open3D uses y down, z out of camera, x right)
    # so to get the world-to-Open3D camera transformation, we need to flip the y and z axes.
    extrinsic_matrix = np.array(
        [[1.0, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    ) @ invert_se3(camera_pose)
    extrinsic_matrix = open3d.core.Tensor(extrinsic_matrix)

    # also, the principal point for y is now measured from the top of the image, not the bottom
    intrinsic_matrix[1, 2] = opts.img_height - intrinsic_matrix[1, 2]
    px = round(intrinsic_matrix[0, 2])
    py = round(intrinsic_matrix[1, 2])
    intrinsic_matrix = open3d.core.Tensor(intrinsic_matrix)

    rays = open3d.t.geometry.RaycastingScene.create_rays_pinhole(
        intrinsic_matrix, extrinsic_matrix, opts.img_width, opts.img_height
    ).numpy()

    # normalise ray direction vectors to unit length
    rays[:, :, 3:] = rays[:, :, 3:] / np.linalg.norm(
        rays[:, :, 3:], axis=-1, keepdims=True
    )

    ray_directions = rays[:, :, 3:]
    center_ray_direction = ray_directions[py, px]

    tensor_rays = open3d.core.Tensor(rays)

    distances_to_hit = raycaster.cast_rays(tensor_rays)["t_hit"].numpy()
    depth_values = distances_to_hit * np.einsum(
        "i,hwi->hw", center_ray_direction, ray_directions
    )  # project the ray direction onto the center ray direction to get the depth values

    # set depth values to 0 for pixels that are not hit
    depth_values[depth_values > 100] = 0

    return depth_values


def load_raycaster(scene_dir):
    all_mesh_files = os.listdir(os.path.join(scene_dir, "region_segmentations"))
    all_regions = list(
        set([f.split(".")[0] for f in all_mesh_files if f.startswith("region")])
    )
    all_regions.sort()

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

    # For some reason open3d has more than one TriangleMesh shape
    raycaster_mesh = open3d.t.geometry.TriangleMesh(
        open3d.core.Tensor(np.asarray(scene_mesh.vertices).astype(np.float32)),
        open3d.core.Tensor(np.asarray(scene_mesh.triangles).astype(np.float32)),
    )

    raycaster = open3d.t.geometry.RaycastingScene()
    raycaster.add_triangles(raycaster_mesh)

    return raycaster


def read_cam_parameters(camera_parameter_file_path):
    with open(camera_parameter_file_path, "r") as f:
        params_file_lines = f.readlines()

    current_intrinsics = None
    image_files = []
    depth_files = []
    camera_poses = []
    intrinsic_matrices = []

    for line in params_file_lines:
        if line.startswith("intrinsics"):
            intrinsics_line = [
                i for i in line.strip().split(" ") if not (i.isspace() or len(i) == 0)
            ]

            current_intrinsics = np.array([float(i) for i in intrinsics_line[1:]])
            current_intrinsics = rearrange(current_intrinsics, "(h w) -> h w", h=3, w=3)
            # adjust for larger pixels
            current_intrinsics[0] *= opts.img_width / 1280
            current_intrinsics[1] *= opts.img_height / 1024

        elif line.startswith("scan"):
            scan_line = line.split(" ")[1:]

            img_file_name = scan_line[1]
            depth_file_name = scan_line[0]
            cam_pose = np.array([float(i) for i in scan_line[2:]])
            cam_pose = rearrange(cam_pose, "(h w) -> h w", h=4, w=4)

            image_files.append(img_file_name)
            depth_files.append(depth_file_name)
            camera_poses.append(cam_pose)
            intrinsic_matrices.append(current_intrinsics)

        else:
            continue

    return image_files, depth_files, camera_poses, intrinsic_matrices


def precompute_color_and_depth(scene, scene_dir, opts):
    old_color_imgs_path = os.path.join(scene_dir, "undistorted_color_images")
    old_depth_path = os.path.join(scene_dir, "undistorted_depth_images")

    if not os.path.exists(os.path.join(scene_dir, opts.color_dir_name)):
        os.makedirs(os.path.join(scene_dir, opts.color_dir_name))
    if not os.path.exists(os.path.join(scene_dir, opts.depth_dir_name)):
        os.makedirs(os.path.join(scene_dir, opts.depth_dir_name))

    # Load raycaster for scene
    raycaster = load_raycaster(scene_dir)

    # load camera parameters
    camera_parameter_file_path = os.path.join(
        scene_dir, "undistorted_camera_parameters", "{}.conf".format(scene)
    )
    image_files, depth_files, camera_poses, intrinsic_matrices = read_cam_parameters(
        camera_parameter_file_path
    )

    n_times_meshes_were_used = 0
    n_times_inpainting_was_used = 0
    n_images_skipped = 0

    for i in tqdm(
        range(len(camera_poses)),
        desc="Precomputing images and depths for scene {}".format(scene),
    ):
        camera_pose = camera_poses[i]
        intrinsic_matrix = intrinsic_matrices[i]
        image_file_name = image_files[i]
        depth_file_name = depth_files[i]

        # Load ground-truth color image
        img = cv2.imread(os.path.join(old_color_imgs_path, image_file_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (opts.img_width, opts.img_height))

        # Load ground-truth depth image
        depths = cv2.imread(
            os.path.join(old_depth_path, depth_file_name), cv2.IMREAD_ANYDEPTH
        ).astype(np.float32)
        depths = depths / 4000.0  # scale to metres; 0.25mm per unit
        depths = cv2.resize(
            depths,
            (opts.img_width, opts.img_height),
            interpolation=cv2.INTER_NEAREST_EXACT,
        )

        # fill in bad or missing depth values
        bad_depths = bad_depth_values(depths)
        n_bad_gt_depths = np.sum(bad_depths)

        if n_bad_gt_depths > 0:  # fill in bad depth values using 3D mesh
            depths_from_mesh = calc_depths_from_raycaster(
                raycaster, camera_pose, intrinsic_matrix, opts
            )
            depths[bad_depths] = depths_from_mesh[bad_depths]

            bad_depths = bad_depth_values(depths)
            n_bad_depths_v1 = np.sum(bad_depths)
            print(
                "Reduced number of bad depths from {} to {} using the 3D Mesh".format(
                    n_bad_gt_depths, n_bad_depths_v1
                )
            )
            n_times_meshes_were_used += 1
            if n_bad_depths_v1 >= 0.2 * opts.img_width * opts.img_height:
                n_images_skipped += 1
                # too many bad depths to fill in with inpainting
                print(
                    "Skipping image {} in scene {} due to too many bad depths".format(
                        image_file_name, scene
                    )
                )
                continue

            elif n_bad_depths_v1 > 0:  # fill in bad depth values using inpainting
                n_times_inpainting_was_used += 1

                depths = inpaint_depth(depths, img)

                bad_depths = bad_depth_values(depths)
                n_bad_depths_final = np.sum(bad_depths)
                print(
                    "Reduced number of bad depths from {} to {} using inpainting".format(
                        n_bad_depths_v1, n_bad_depths_final
                    )
                )

        assert np.sum(bad_depth_values(depths)) == 0

        np.save(os.path.join(scene_dir, opts.color_dir_name, image_file_name[:-4]), img)
        np.save(
            os.path.join(scene_dir, opts.depth_dir_name, depth_file_name[:-4]), depths
        )

    print("Finished precomputing scene {}".format(scene))
    print(
        "Meshes were used to fill in bad depths {} times".format(
            n_times_meshes_were_used
        )
    )
    print(
        "Inpainting was used to fill in bad depths {} times".format(
            n_times_inpainting_was_used
        )
    )
    print("{} images were skipped due to too many bad depths".format(n_images_skipped))


def main(opts):
    for scene in ALL_SCENES:
        print(f"Processing scene {scene}")
        scene_dir = os.path.join(opts.raw_dataset_dir, scene)

        # unzip the files
        if "region_segmentations.zip" in os.listdir(scene_dir):
            with zipfile.ZipFile(
                os.path.join(scene_dir, "region_segmentations.zip"), "r"
            ) as zip_ref:
                zip_ref.extractall(opts.raw_dataset_dir)

            os.remove(os.path.join(scene_dir, "region_segmentations.zip"))

        if "undistorted_camera_parameters.zip" in os.listdir(scene_dir):
            with zipfile.ZipFile(
                os.path.join(scene_dir, "undistorted_camera_parameters.zip"), "r"
            ) as zip_ref:
                zip_ref.extractall(opts.raw_dataset_dir)

            os.remove(os.path.join(scene_dir, "undistorted_camera_parameters.zip"))

        if "undistorted_color_images.zip" in os.listdir(scene_dir):
            with zipfile.ZipFile(
                os.path.join(scene_dir, "undistorted_color_images.zip"), "r"
            ) as zip_ref:
                zip_ref.extractall(opts.raw_dataset_dir)

            os.remove(os.path.join(scene_dir, "undistorted_color_images.zip"))

        if "undistorted_depth_images.zip" in os.listdir(scene_dir):
            with zipfile.ZipFile(
                os.path.join(scene_dir, "undistorted_depth_images.zip"), "r"
            ) as zip_ref:
                zip_ref.extractall(opts.raw_dataset_dir)

            os.remove(os.path.join(scene_dir, "undistorted_depth_images.zip"))

        precompute_color_and_depth(scene, scene_dir, opts)

    print("Done!")


if __name__ == "__main__":
    parser = configargparse.ArgumentParser()
    parser.add_argument(
        "--raw_dataset_dir",
        type=str,
        default="data/matterport3d",
    )
    parser.add_argument("--img_width", type=int, default=320)
    parser.add_argument("--img_height", type=int, default=256)
    parser.add_argument("--color_dir_name", type=str, default="rgb")
    parser.add_argument("--depth_dir_name", type=str, default="depth")
    opts = parser.parse_args()
    main(opts)
