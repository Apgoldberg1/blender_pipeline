import json
from pathlib import Path
from typing import Literal, Tuple, Dict
import tqdm
import os

import matplotlib.pyplot as plt
import torch
import numpy as np

from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.cameras.cameras import Cameras


def _setup(config: str, camera_poses: str):
    """
    config: path to model training run config
    camera_poses: str path to transforms.json formatted camera poses
    """

    _, pipeline, _, _ = eval_setup(
        config_path=Path(config),
        test_mode="test",
    )

    camera_poses_path = "test_poses.json"
    with open(camera_poses_path, "r") as f:
        camera_poses = json.load(f)

    num_cameras = len(camera_poses["frames"])
    cameras = []

    for i in range(num_cameras):
        camera_to_worlds = torch.tensor(camera_poses["frames"][i]["transform_matrix"][:3]).view(1, 3, 4)
        fx, fy = torch.tensor(camera_poses["fl_x"]).view(1, 1), torch.tensor(camera_poses["fl_y"]).view(1, 1)
        cx, cy = torch.tensor(camera_poses["cx"]).view(1, 1), torch.tensor(camera_poses["cy"]).view(1, 1)
        w, h = torch.tensor(camera_poses["w"], dtype=torch.int).view(1, 1), torch.tensor(camera_poses["h"], dtype=torch.int).view(1, 1)

        camera = Cameras(camera_to_worlds, fx, fy, cx, cy, w, h).to(pipeline.device)
        cameras.append(camera)

        return pipeline, cameras


def render_depth_imgs(config: str, camera_poses: str):
    pipeline, cameras = _setup(config, camera_poses)

    output_ims = []

    with torch.no_grad():
        for i in range(len(cameras)):
            output_ims.append(pipeline.model.get_outputs(cameras[i])["rgb"].cpu())

    return output_ims

def render_and_save_depth_imgs(config: str, camera_poses: str, output_dir: str):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    imgs = render_depth_imgs(config, camera_poses)

    print(f"saving {len(imgs)} imgs to {output_dir}")

    for i, img in enumerate(imgs):
        img_path = os.path.join(output_dir, f"img_{i}.png")
        plt.imsave(img_path, img.numpy())
    

if __name__ == "__main__":
    config_path = "outputs/glass_400/splatfacto/2024-02-26_222509/config.yml"
    camera_poses = "test_poses.json"
    output_dir = "test_render_depth"

    depth_ims = render_depth_imgs(config_path, camera_poses)

    plt.imshow(depth_ims[1])
    plt.show()


