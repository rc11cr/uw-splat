# Script that renders underwater image given RGB+D and backscatter / attenuation parameters

import math
import os
import time
from os import makedirs
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from tqdm import tqdm

from scene import Scene
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render, render_depth, GaussianModel
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.loss_utils import ssim
from utils.graphics_utils import getProjectionMatrix
from scene.cameras import MiniCam

from lpipsPyTorch import lpips

# ---------  UNDERWATER MODELS (MLP)  ---------
from underwater.models import (
    BackscatterMLP,
    AttenuateMLP,
)

# ---------- 可选：物理参数版“加水”工具（用不到可以忽略） ----------
WATER_BETA_D = [2.6, 2.4, 1.8]
WATER_BETA_B = [1.9, 1.7, 1.4]
WATER_B_INF = [0.07, 0.2, 0.39]


def render_uw_analytic(rgb, d):
    """
    用固定 beta 参数把一个“干净”场景 (rgb, depth) 渲染成合成水下图像。
    rgb: [3,H,W], d: [1,H,W] in [0,1]
    """
    J = rgb

    tmp = torch.from_numpy(np.array(WATER_BETA_D)).view((3, 1, 1, 1)).float().cuda()
    at_exp_coeff = -1.0 * torch.nn.functional.conv2d(d, tmp)
    attenuation = torch.exp(at_exp_coeff)

    tmp2 = torch.from_numpy(np.array(WATER_BETA_B)).view((3, 1, 1, 1)).float().cuda()
    bs_exp_coeff = -1.0 * torch.nn.functional.conv2d(d, tmp2)
    backscatter = 1 - torch.exp(bs_exp_coeff)

    tmp3 = torch.from_numpy(np.array(WATER_B_INF)).view(3, 1, 1).float().cuda()
    I = J * attenuation + tmp3 * backscatter
    return I


# ============================================================
#          Dark Channel / Atmospheric Light Estimation
# ============================================================

def dark_channel_estimate(rgb):
    """
    Dark Channel Prior 计算，用于估计大气/水体光照颜色。
    rgb: [3, H, W] 或 [B, 3, H, W]
    """
    patch_size = 41
    padding = patch_size // 2
    rgb_max = torch.nn.MaxPool3d(
        kernel_size=(3, patch_size, patch_size),
        stride=1,
        padding=(0, padding, padding),
    )
    if len(rgb.size()) == 3:
        # CHW -> 1CHW
        dcp = torch.abs(rgb_max(-rgb.unsqueeze(0)))
    else:
        dcp = torch.abs(rgb_max(-rgb))
    return dcp.squeeze()


def estimate_atmospheric_light(rgb):
    """
    从 dark channel 估计水体/大气光颜色（B_inf 的 target）

    rgb: [3, H, W]
    return: [3] 向量
    """
    dcp = dark_channel_estimate(rgb)

    flat_dcp = torch.flatten(dcp)
    flat_r = torch.flatten(rgb[0])
    flat_g = torch.flatten(rgb[1])
    flat_b = torch.flatten(rgb[2])

    # 取最亮的 0.1% dark-channel 像素
    k = math.ceil(0.001 * len(flat_dcp))
    vals, idxs = torch.topk(flat_dcp, k)

    median_val = torch.median(vals)
    median_idxs = torch.where(vals == median_val)
    color_idxs = idxs[median_idxs]

    atmospheric_light = torch.stack([
        torch.mean(flat_r[color_idxs]),
        torch.mean(flat_g[color_idxs]),
        torch.mean(flat_b[color_idxs]),
    ])

    return atmospheric_light


# ============================================================
#                DATASET (TRAIN / TEST) 渲染 + 评估
# ============================================================

def render_set_dataset(
    model_path,
    name,
    iteration,
    views,
    gaussians,
    pipeline,
    render_background,
    do_seathru: bool = False,
    add_water: bool = False,
    learned_bg=None,
    bs_model=None,
    at_model=None,
    save_as_jpeg: bool = False,
):
    """
    对 train 或 test 的 camera 集合进行渲染：
    - 保存 render / depth / alpha / (可选) with_water 等图片
    - 直接在内存里用 GT 计算 PSNR / SSIM / LPIPS 并 print
    """

    model_path = Path(model_path)

    if do_seathru:
        assert bs_model is not None
        assert at_model is not None
        no_water_dir = model_path / name / "no_water"
        backscatter_dir = model_path / name / "backscatter"
        attenuation_dir = model_path / name / "attenuation"
        with_water_dir = model_path / name / "with_water"
        makedirs(no_water_dir, exist_ok=True)
        makedirs(backscatter_dir, exist_ok=True)
        makedirs(attenuation_dir, exist_ok=True)
        makedirs(with_water_dir, exist_ok=True)

    if learned_bg is not None:
        bg_dir = model_path / name / "bg"
        makedirs(bg_dir, exist_ok=True)

    if add_water:
        water_dir = model_path / name / "add_water"
        makedirs(water_dir, exist_ok=True)

    render_dir = model_path / name / "render"
    depth_dir = model_path / name / "depth"
    alpha_dir = model_path / name / "alpha"
    makedirs(render_dir, exist_ok=True)
    makedirs(depth_dir, exist_ok=True)
    makedirs(alpha_dir, exist_ok=True)

    timings_3dgs = []
    timings_seathru = []

    # 指标
    ssims = []
    psnrs = []
    lpipss = []

    for idx, view in enumerate(tqdm(views, desc=f"Rendering {name}")):
        start_time = time.time()
        render_pkg = render(view, gaussians, pipeline, render_background)
        rendered_image, image_alpha = render_pkg["render"], render_pkg["alpha"]
        render_depth_pkg = render_depth(view, gaussians, pipeline, render_background)
        depth_image = render_depth_pkg["render"][0].unsqueeze(0)  # [1,H,W]
        depth_image = depth_image / image_alpha
        end_3dgs = time.time()

        # depth 清理 + 归一化
        if torch.any(torch.logical_or(torch.isnan(depth_image), torch.isinf(depth_image))):
            valid_depth_vals = depth_image[
                torch.logical_not(torch.logical_or(torch.isnan(depth_image), torch.isinf(depth_image)))
            ]
            if len(valid_depth_vals) == 0:
                print(f"[{name}] everything is nan {view.image_name}")
                not_nan_max = 100.0
            else:
                not_nan_max = torch.max(valid_depth_vals).item()
            depth_image = torch.nan_to_num(depth_image, not_nan_max, not_nan_max)

        if depth_image.min() != depth_image.max():
            depth_image = (depth_image - depth_image.min()) / (depth_image.max() - depth_image.min())
        else:
            depth_image = depth_image / depth_image.max()

        if learned_bg is not None:
            bg_image = torch.sigmoid(learned_bg).reshape(3, 1, 1) * (1 - image_alpha)
            image = rendered_image + bg_image
            torchvision.utils.save_image(bg_image, bg_dir / f"{view.image_name}.png")
        else:
            image = rendered_image

        if add_water:
            # 用 analytic 参数往 GT 图像里“硬加水”，主要用于可视化
            added_uw_image = render_uw_analytic(view.original_image, depth_image)
            torchvision.utils.save_image(added_uw_image, water_dir / f"{view.image_name}.png")

        if do_seathru:
            image_batch = torch.unsqueeze(image, dim=0)          # [1,3,H,W]
            depth_image_batch = torch.unsqueeze(depth_image, 0)  # [1,1,H,W]

            attenuation_map_batch = at_model(depth_image_batch)
            backscatter_batch = bs_model(depth_image_batch)
            direct_batch = image_batch * attenuation_map_batch
            underwater_image_batch = torch.clamp(direct_batch + backscatter_batch, 0.0, 1.0)
            end_seathru = time.time()
            timings_seathru.append(end_seathru - start_time)

            # 保存各种 map
            torchvision.utils.save_image(image, no_water_dir / f"{view.image_name}.png")
            torchvision.utils.save_image(backscatter_batch.squeeze(), backscatter_dir / f"{view.image_name}.png")
            torchvision.utils.save_image(attenuation_map_batch.squeeze(), attenuation_dir / f"{view.image_name}.png")
            if save_as_jpeg:
                torchvision.utils.save_image(
                    underwater_image_batch.squeeze(), with_water_dir / f"{view.image_name}.jpg"
                )
            else:
                torchvision.utils.save_image(
                    underwater_image_batch.squeeze(), with_water_dir / f"{view.image_name}.png"
                )
            final_image = underwater_image_batch.squeeze(0)  # [3,H,W]
        else:
            # 只 3DGS
            final_image = image
            if save_as_jpeg:
                torchvision.utils.save_image(rendered_image, render_dir / f"{view.image_name}.jpg")
            else:
                torchvision.utils.save_image(rendered_image, render_dir / f"{view.image_name}.png")

        # 保存 depth / alpha
        torchvision.utils.save_image(depth_image, depth_dir / f"{view.image_name}.png")
        torchvision.utils.save_image(image_alpha, alpha_dir / f"{view.image_name}.png")

        timings_3dgs.append(end_3dgs - start_time)

        # ========== 计算指标（跟 GT 比） ==========
        gt_image = torch.clamp(view.original_image.to("cuda"), 0.0, 1.0)  # [3,H,W]

        pred = final_image.unsqueeze(0)  # [1,3,H,W]
        gt = gt_image.unsqueeze(0)

        ssims.append(ssim(pred, gt))
        psnrs.append(psnr(pred, gt))

        # ---- 这里是修好的 LPIPS 用法（函数调用）----
        pred_lp = pred * 2.0 - 1.0
        gt_lp = gt * 2.0 - 1.0
        lpip_val = lpips(pred_lp, gt_lp, net_type="vgg")  # 返回 tensor
        lpipss.append(lpip_val.mean())

    # --------- 打印时间和指标 ----------
    if do_seathru and len(timings_seathru) > 0:
        print(f"[Seathru-{name}] Average time for {len(views)} images: {np.mean(timings_seathru):.4f}s")
    print(f"[3DGS-{name}] Average time for {len(views)} images: {np.mean(timings_3dgs):.4f}s")

    ssims_t = torch.tensor(ssims).mean().item()
    psnrs_t = torch.tensor(psnrs).mean().item()
    lpips_t = torch.stack(lpipss).mean().item()

    print(f"-----------{name.capitalize()}")
    print("  SSIM ^: {:>12.7f}".format(ssims_t, ".5"))
    print("  PSNR ^: {:>12.7f}".format(psnrs_t, ".5"))
    print("  LPIPS v: {:>12.7f}".format(lpips_t, ".5"))
    print("")

    return {
        "SSIM": ssims_t,
        "PSNR": psnrs_t,
        "LPIPS": lpips_t,
    }


# ============================================================
#        兼容 train.py 的 render_set 包装函数（重点）
# ============================================================

def render_set(
    model_path,
    name,
    iteration,
    views,
    gaussians,
    pipeline,
    render_background,
    do_seathru: bool = False,
    add_water: bool = False,
    add_fog: bool = False,   # 兼容旧接口，这里忽略
    learned_bg=None,
    bs_model=None,
    at_model=None,
    save_as_jpeg: bool = False,
):
    """
    兼容 train.py 原来的 render_set 接口：
    - 忽略 add_fog（因为现在只关心水下，不做雾）
    - 内部调用 render_set_dataset 做真正的渲染 + 指标打印
    """
    return render_set_dataset(
        model_path,
        name,
        iteration,
        views,
        gaussians,
        pipeline,
        render_background,
        do_seathru=do_seathru,
        add_water=add_water,
        learned_bg=learned_bg,
        bs_model=bs_model,
        at_model=at_model,
        save_as_jpeg=save_as_jpeg,
    )


def render_sets_dataset(
    model_params: ModelParams,
    iteration: int,
    pipeline: PipelineParams,
    skip_train: bool,
    skip_test: bool,
    do_seathru: bool,
    add_water: bool,
):
    with torch.no_grad():
        gaussians = GaussianModel(model_params.sh_degree)
        scene = Scene(model_params, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1, 1, 1] if model_params.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        learned_bg = None

        bs_model = None
        at_model = None
        if do_seathru:
            attenuation_model_path = f"{model_params.model_path}/attenuate_{scene.loaded_iter}.pth"
            backscatter_model_path = f"{model_params.model_path}/backscatter_{scene.loaded_iter}.pth"
            print(f"Loading backscatter model {backscatter_model_path}")
            print(f"Loading attenuation model {attenuation_model_path}")
            assert os.path.exists(attenuation_model_path)
            assert os.path.exists(backscatter_model_path)
            at_model = AttenuateMLP()
            bs_model = BackscatterMLP()
            at_model.load_state_dict(torch.load(attenuation_model_path))
            bs_model.load_state_dict(torch.load(backscatter_model_path))
            at_model.cuda().eval()
            bs_model.cuda().eval()

        results = {}

        if not skip_train:
            res_train = render_set_dataset(
                Path(model_params.model_path),
                "train",
                scene.loaded_iter,
                scene.getTrainCameras(),
                gaussians,
                pipeline,
                background,
                do_seathru,
                add_water,
                learned_bg,
                bs_model,
                at_model,
            )
            results["Train"] = res_train

        if not skip_test and model_params.eval:
            res_test = render_set_dataset(
                Path(model_params.model_path),
                "test",
                scene.loaded_iter,
                scene.getTestCameras(),
                gaussians,
                pipeline,
                background,
                do_seathru,
                add_water,
                learned_bg,
                bs_model,
                at_model,
            )
            results["Test"] = res_test

    return results


# ============================================================
#                      RENDER PATH（无指标）
# ============================================================

def render_set_path(
    model_path,
    name,
    iteration,
    views,
    gaussians,
    pipeline,
    render_background,
    do_seathru: bool = False,
    bs_model=None,
    at_model=None,
    render_path_override=None,
):
    """
    根据给定 camera pose 轨迹进行渲染，不算指标（一般是 novel view，没有 GT）。
    """
    if render_path_override is None:
        return

    model_path = Path(model_path)

    if do_seathru:
        assert bs_model is not None
        assert at_model is not None
        backscatter_dir = model_path / name / "backscatter"
        attenuation_dir = model_path / name / "attenuation"
        backscatter_normed_dir = model_path / name / "backscatter_normed"
        attenuation_normed_dir = model_path / name / "attenuation_normed"
        with_water_dir = model_path / name / "with_water"
        makedirs(backscatter_dir, exist_ok=True)
        makedirs(attenuation_dir, exist_ok=True)
        makedirs(backscatter_normed_dir, exist_ok=True)
        makedirs(attenuation_normed_dir, exist_ok=True)
        makedirs(with_water_dir, exist_ok=True)

    render_dir = model_path / name / "render"
    depth_dir = model_path / name / "depth"
    depth_viridis_dir = model_path / name / "depth_viridis"
    alpha_dir = model_path / name / "alpha"
    makedirs(render_dir, exist_ok=True)
    makedirs(depth_dir, exist_ok=True)
    makedirs(depth_viridis_dir, exist_ok=True)
    makedirs(alpha_dir, exist_ok=True)

    just_viridis = False

    view0 = views[0]
    fov_x = view0.FoVx
    fov_y = view0.FoVy
    znear = view0.znear
    zfar = view0.zfar
    width = view0.image_width
    height = view0.image_height

    for frame_num, T_world_cam in enumerate(tqdm(render_path_override, desc=f"Rendering path {name}")):
        frame_num_str = f"{frame_num:04d}"
        T_world_cam = torch.Tensor(T_world_cam).cuda()
        T_cam_world = T_world_cam.inverse()
        world_view_transform = T_cam_world.T

        projection_matrix = getProjectionMatrix(
            znear=znear, zfar=zfar, fovX=fov_x, fovY=fov_y
        ).transpose(0, 1).cuda()
        full_proj_transform = world_view_transform.unsqueeze(0).bmm(
            projection_matrix.unsqueeze(0)
        ).squeeze(0)

        render_view = MiniCam(
            width=width,
            height=height,
            fovy=fov_y,
            fovx=fov_x,
            znear=znear,
            zfar=zfar,
            world_view_transform=world_view_transform,
            full_proj_transform=full_proj_transform,
        )

        render_pkg = render(render_view, gaussians, pipeline, render_background)
        rendered_image, image_alpha = render_pkg["render"], render_pkg["alpha"]

        render_depth_pkg = render_depth(render_view, gaussians, pipeline, render_background)
        depth_image = render_depth_pkg["render"][0].unsqueeze(0)
        depth_image = depth_image / image_alpha

        if torch.any(torch.logical_or(torch.isnan(depth_image), torch.isinf(depth_image))):
            valid_depth_vals = depth_image[
                torch.logical_not(torch.logical_or(torch.isnan(depth_image), torch.isinf(depth_image)))
            ]
            if len(valid_depth_vals) == 0:
                print(f"[path] everything is nan")
                not_nan_max = 100.0
            else:
                not_nan_max = torch.max(valid_depth_vals).item()
            depth_image = torch.nan_to_num(depth_image, not_nan_max, not_nan_max)
        if depth_image.min() != depth_image.max():
            depth_image = (depth_image - depth_image.min()) / (depth_image.max() - depth_image.min())
        else:
            depth_image = depth_image / depth_image.max()

        normalized_depth_image = depth_image / depth_image.max()

        if do_seathru:
            image_batch = torch.unsqueeze(rendered_image, dim=0)
            depth_image_batch = torch.unsqueeze(depth_image, dim=0)

            attenuation_map = at_model(depth_image_batch)
            direct = image_batch * attenuation_map
            attenuation_map_normalized = attenuation_map / torch.max(attenuation_map)

            backscatter = bs_model(depth_image_batch)
            backscatter_normalized = backscatter / torch.max(backscatter)

            underwater_image = torch.clamp(direct + backscatter, 0.0, 1.0)

            underwater_image = underwater_image.squeeze()
            if not just_viridis:
                torchvision.utils.save_image(underwater_image, with_water_dir / f"render_{frame_num_str}.png")
                torchvision.utils.save_image(attenuation_map, attenuation_dir / f"render_{frame_num_str}.png")
                torchvision.utils.save_image(
                    attenuation_map_normalized, attenuation_normed_dir / f"render_{frame_num_str}.png"
                )
                torchvision.utils.save_image(backscatter, backscatter_dir / f"render_{frame_num_str}.png")
                torchvision.utils.save_image(
                    backscatter_normalized, backscatter_normed_dir / f"render_{frame_num_str}.png"
                )

        if not just_viridis:
            torchvision.utils.save_image(rendered_image, render_dir / f"render_{frame_num_str}.png")
            torchvision.utils.save_image(normalized_depth_image, depth_dir / f"render_{frame_num_str}.png")
            torchvision.utils.save_image(image_alpha, alpha_dir / f"render_{frame_num_str}.png")
        plt.imsave(
            depth_viridis_dir / f"render_{frame_num_str}.png",
            normalized_depth_image.detach().cpu().numpy().squeeze(),
        )


def render_sets_path(
    model_params: ModelParams,
    iteration: int,
    pipeline: PipelineParams,
    do_seathru: bool,
    render_path: str,
    out_folder_name: str,
):
    with torch.no_grad():
        gaussians = GaussianModel(model_params.sh_degree)
        scene = Scene(model_params, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1, 1, 1] if model_params.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        bs_model = None
        at_model = None
        if do_seathru:
            attenuation_model_path = f"{model_params.model_path}/attenuate_{scene.loaded_iter}.pth"
            backscatter_model_path = f"{model_params.model_path}/backscatter_{scene.loaded_iter}.pth"
            print(f"Loading backscatter model {backscatter_model_path}")
            print(f"Loading attenuation model {attenuation_model_path}")
            assert os.path.exists(attenuation_model_path)
            assert os.path.exists(backscatter_model_path)
            at_model = AttenuateMLP()
            bs_model = BackscatterMLP()
            at_model.load_state_dict(torch.load(attenuation_model_path))
            bs_model.load_state_dict(torch.load(backscatter_model_path))
            at_model.cuda().eval()
            bs_model.cuda().eval()

        render_path_poses = None
        if render_path is not None:
            render_path_poses = np.load(render_path)
            if "stn" in render_path:
                render_path_poses = render_path_poses @ np.diag([1, -1, -1, 1])

        render_set_path(
            Path(model_params.model_path),
            out_folder_name,
            scene.loaded_iter,
            scene.getTrainCameras(),  # 用 train 的 intrinsics
            gaussians,
            pipeline,
            background,
            do_seathru,
            bs_model,
            at_model,
            render_path_poses,
        )


# ============================================================
#                          MAIN
# ============================================================

if __name__ == "__main__":
    parser = ArgumentParser(description="Underwater rendering / evaluation script")
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--seathru", action="store_true")
    parser.add_argument("--add_water", action="store_true")      # 只在 dataset 模式用 analytic 加水
    parser.add_argument("--render_path", default=None, type=str) # 不为 None 时走相机轨迹模式
    parser.add_argument("--out_folder_name", default="icra", type=str)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # 防止旧 cfg_args 里没有 render_path 字段时报 AttributeError
    if not hasattr(args, "render_path"):
        args.render_path = None

    safe_state(args.quiet, seed=0)

    if args.render_path is None:
        # 数据集 train / test 渲染 + 直接 print PSNR / SSIM / LPIPS
        render_sets_dataset(
            model.extract(args),
            args.iteration,
            pipeline.extract(args),
            args.skip_train,
            args.skip_test,
            args.seathru,
            args.add_water,
        )
    else:
        # 沿渲染轨迹输出视频帧，不算指标
        render_sets_path(
            model.extract(args),
            args.iteration,
            pipeline.extract(args),
            args.seathru,
            args.render_path,
            args.out_folder_name,
        )