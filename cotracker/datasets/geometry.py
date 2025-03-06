import torch
import numpy as np
from loguru import logger
import matplotlib.pyplot as plt
from einops import rearrange, repeat
from torch.nn import functional as F
from kornia.utils import create_meshgrid
from cotracker.datasets.utils import CoTrackerData
from cotracker.models.core.model_utils import get_query_ponts
def plot(src_pts, vector):
    """
    Only for debug visualization.
    """
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.quiver(src_pts[:, 0],src_pts[:, 1], src_pts[:,2], vector[:,0], vector[:,1], vector[:,2], normalize=False, color='r', arrow_length_ratio=0.15)

    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')

    # ax.set_title('b-vectors on unit sphere')

    plt.savefig('test.png')



@torch.no_grad()
def warp_source_views(
    src_points,
    src_depth_map,
    src_intrinsic,
    src_extrin,
    src_origin_img_size,
    dst_intrinsic,
    dst_extrin,
    dst_depth_maps,
    dst_origin_imgs_sizes,
    depth_consistency_thres=0.2,
    cycle_reproj_distance_thres=5, # pixel
    border_thres=2,
):
    """
    Warp source view points to multiple query views and check depth consistency.

    Args:
        src_view_points: B * n_pts * 2 - <x, y>
        src_view_depth_map: B * H * W
        src_intrinsic: B * 3 * 3
        src_extrin: B * 4 * 4, from world to cam
        src_origin_img_size: B * 2, - <h, w>
        dst_intrinsic: B * n_dst * 3 * 3
        dst_depth_maps: B * n_dst * H * W
        dst_mask: B * n_ds * H * W
        dst_imgs_sizes: B * n_dst * 2
        dst_extrin: B * n_dst * 4 * 4, from world to cam

    Return:
        valid_mask: B * n_dst * n_pts
        dst_pts: B * n_dst * n_pts * 2
        world_points: B * n_pts * 3
    """
    src_points = src_points.round().long()
    device = src_points.device
    B, N_pts = src_points.shape[:2]
    _, N_dst = dst_intrinsic.shape[:2]

    # Sample depth, get calculable_mask on depth != 0
    src_points_depth = torch.stack(
        [
            src_depth_map[i, src_points[i, :, 1], src_points[i, :, 0]]
            for i in range(src_points.shape[0])
        ],
        dim=0,
    )  # B * N_pts
    nonzero_mask = src_points_depth != 0  # B * N_pts
    src_border_mask = (
        (src_points[:, :, 0] > border_thres)
        * (src_points[:, :, 1] > border_thres)
        * (src_points[:, :, 0] < (src_origin_img_size[:, None, 1] - border_thres))
        * (src_points[:, :, 1] < (src_origin_img_size[:, None, 0] - border_thres))
    ) # B * N_pts

    # Unproject:
    src_points_h = (
        torch.cat([src_points, torch.ones((B, N_pts, 1), device=device)], dim=-1)
        * src_points_depth[..., None]
    )  # B * N_pts * 3
    src_points_cam = src_intrinsic.inverse() @ src_points_h.transpose(
        2, 1
    )  # B * 3 * N_pts

    # From source cam to world:
    src_pose = src_extrin.inverse()
    world_points = (
        src_pose[:, :3, :3] @ src_points_cam + src_pose[:, :3, [3]]
    )  # B * 3 * N_pts

    # Transform to dst views:
    dst_pts_cam = (
        dst_extrin[..., :3, :3] @ world_points[:, None] + dst_extrin[..., :3, [3]]
    )  # B * N_dst * 3 * N_pts

    # Project:
    dst_pts_h = (dst_intrinsic @ dst_pts_cam).transpose(3, 2)  # B * N_dst * N_pts * 3
    proj_depth = dst_pts_h[..., 2]  # B * N_dst * N_pts * 1
    dst_pts = dst_pts_h[..., :2] / (dst_pts_h[..., [2]] + 1e-4)  # B * N_dst * N_pts * 2

    # Covis check:
    h, w = dst_depth_maps.shape[-2:]
    covisible_mask = (
        (dst_pts[..., 0] > border_thres)
        * (dst_pts[..., 0] < dst_origin_imgs_sizes[..., None, 1] - border_thres)
        * (dst_pts[..., 1] > border_thres)
        * (dst_pts[..., 1] < dst_origin_imgs_sizes[..., None, 0] - border_thres)
    )  # B * N_dst * N_pts
    dst_pts[~covisible_mask, :] = 0  # B * N_dst * N_pts * 2

    # Depth consistency check:
    dst_pts_long = (dst_pts.long()).view(B * N_dst, N_pts, 2)
    dst_depth_maps = rearrange(dst_depth_maps, "b n h w -> (b n) h w")
    sampled_depth = torch.stack(
        [
            dst_depth_maps[i, dst_pts_long[i, :, 1], dst_pts_long[i, :, 0]]
            for i in range(dst_pts_long.shape[0])
        ],
        dim=0,
    )
    sampled_depth = rearrange(
        sampled_depth, "(b n) m -> b n m", b=B
    )  # B * N_dst * N_pts
    consistency_mask = (
        (sampled_depth - proj_depth) / (sampled_depth + 1e-4)
    ).abs() < depth_consistency_thres

    # Back project to src view for cycle check:
    dst_pts_h = torch.cat([dst_pts, torch.ones((B, N_dst, N_pts, 1), device=device)], dim=-1) * sampled_depth[..., None] # B * N_dst * N_pts * 3
    dst_pts_cam = dst_intrinsic.inverse() @ dst_pts_h.transpose(2, 3)
    dst_pose = dst_extrin.inverse()
    world_points_cycle_back = dst_pose[:, :, :3, :3] @ dst_pts_cam + dst_pose[:, :, :3, [3]]
    src_warp_back_cam = src_extrin[:, None, :3, :3] @ world_points_cycle_back + src_extrin[:, None, :3, [3]]
    src_warp_back_h = (src_intrinsic @ src_warp_back_cam).transpose(2, 3)
    src_back_proj_depth = src_warp_back_h[..., 2]
    src_back_proj_pts = src_warp_back_h[..., :2] / (src_warp_back_h[..., [2]] + 1e-4)
    cycle_reproj_distance_mask = (torch.linalg.norm(src_back_proj_pts - src_points[:, None], dim=-1)) < cycle_reproj_distance_thres
    cycle_depth_distance_mask = ((src_back_proj_depth - src_points_depth[:, None]).abs() / (src_points_depth[:, None] + 1e-4)) < depth_consistency_thres

    valid_mask = (
        nonzero_mask[:, None] * src_border_mask[:, None] * covisible_mask * consistency_mask * cycle_reproj_distance_mask * cycle_depth_distance_mask
    )  # B * N_dst * N_pts
    """
    # Get absolute scale of each points:
    dst_scales_absolute = dst_intrinsic[:, :, 0, 0][..., None] / (proj_depth + 1e-4) # B * N_dst * N_pts
    src_scales_absolute = src_intrinsic[:, 0, 0][..., None] / (src_points_depth + 1e-4) # B * N_pts
    scale_absolute = torch.cat([src_scales_absolute[:, None], dst_scales_absolute], dim=1) # B * N_view * N_pts

    # Get relative view points infos:
    relative_pose = src_extrin[:, None] @ dst_extrin.inverse()# B * N_dst * 4 * 4
    t = relative_pose[..., :3, 3] # B * N_dst * 3, from src camera to dst camera
    
             /             \
            /               \
         f /                 \ a
          /                   \
         / alpha         beta  \
        /_______________________\
    src_view        t         dst_view;  view_point_vector is encoded by the t_norm and the gamma
    
    # plot(src_pts=np.zeros((t.shape[1], 3)), vector=t[0].cpu().numpy())
    f = repeat(src_points_cam.transpose(1,2), 'b n_track c -> b n_view n_track c', n_view=N_dst) # B * N_dst * N_track * 3
    t = repeat(t, 'b n_view c -> b n_view n_track c', n_track=N_pts) # B * N_dst * N_track * 3
    a = f - t # B * N_track * N_dst * 3
    f_norm, t_norm, a_norm = map(lambda x: torch.linalg.norm(x, dim=-1, keepdim=True), [f, t, a])
    alpha = torch.arccos(torch.einsum('bntc,bntc->bnt', f, t)[..., None] / (f_norm * t_norm + 1e-4))
    beta = torch.arccos(torch.einsum('bntc,bntc->bnt', a, -1 * t)[..., None] / (a_norm * t_norm + 1e-4))
    gamma = torch.arccos(torch.zeros(1, device=a.device)).item() * 2  - alpha - beta
    view_point_vector = (t / (t_norm + 1e-4)) * gamma # B * N_dst * N_track * 3
    view_point_vector = torch.cat([torch.zeros((B, 1, N_pts, 3), device=view_point_vector.device), view_point_vector], dim=1) # B * N_view * N_track * 3
    return valid_mask, dst_pts, world_points.transpose(2, 1), scale_absolute, view_point_vector
    """
    return valid_mask, dst_pts, world_points.transpose(2, 1)

@torch.no_grad()
def mask_grid_pts_at_padded_regions(grid_pt, mask, scale=8):
    """
    For megadepth dataset, zero-padding exists in images
    mask: B * N * H * W (in resized solution)
    grid_pt: B * N * n_grid * 2
    """
    if scale != 1:
        mask = F.interpolate(
            mask, scale_factor=1 / scale, mode="nearest", recompute_scale_factor=False
        )
    mask = repeat(mask, "b n h w -> b n (h w) c", c=2)
    grid_pt[~mask.bool()] = 0
    return grid_pt, mask

@torch.no_grad()
def dense_grid_spv(data):
    """
    Input:
    "images": B * N * 1 * H * W
    Output:
    "points": b_id, n_id, x, y
    "tracks": N_track * track_length, indices of points, where the first is the reference view

    Then input to fineprocess:
    N_track: track_length * WW * C features,
    attention is performed:
    N_track: 1 * WW * C <-> (track_length - 1) * WW * C
    """
    grid_scale = 1
    device = data["depth"].device
    track_length_tolerance = 0
    # Generate grid coordinates:
    B, N, _, H, W = data["video"].shape
    

    scale = (
        grid_scale * data["scales"][..., None, [1, 0]]
        if "scales" in data
        else grid_scale
    )  # B * N_view * 1 * 2

    """
    h_coarse, w_coarse = map(lambda x: x // grid_scale, [H, W])
    grid_coord_c = create_meshgrid(
        h_coarse, w_coarse, normalized_coordinates=False, device=device
    ).reshape(1, 1, h_coarse * w_coarse, 2)
    """
    grid_coord_c, sample_mask = get_query_ponts(data['video'][:,0], max_query_num=data['max_queries'])
    # logger.info(f"shape of grid_coord_c {grid_coord_c.shape} and scale shape {scale.shape}")
    grid_coord_c = grid_coord_c.unsqueeze(0).to(device)  # 1 * 1 * n_points * 2
    # grid_coord_c = grid_coord_c * scale  # B * N * n_points * 2, in original image scale
    # Warp reference view to query views:
    # NOTE: no need mutual nearest neighbour
    query_coords = grid_coord_c.to(device)  # [B, n_pts, 2]
    # query_coords = grid_coord_c[:,0]  # [B, n_pts, 2]
    # Mask padded regions:
    if "masks" in data:
        grid_coord_c, pad_mask = mask_grid_pts_at_padded_regions(
            grid_coord_c, data["masks"], scale=grid_scale
        ) 
        pad_mask= pad_mask[:, 0, :, 0] # Mask: [B, n_pts], for first image
    else:
        pad_mask = None

    valid_mask, warpped_pts, world_points = warp_source_views(
        src_points=query_coords,
        src_depth_map=data["depth"][:, 0],
        src_intrinsic=data["intrinsics"][:, 0],
        src_extrin=data["extrinsics"][:, 0],
        src_origin_img_size=data['original_hw'][:, 0],
        dst_intrinsic=data["intrinsics"][:, 1:],
        dst_extrin=data["extrinsics"][:, 1:],
        dst_depth_maps=data["depth"][:, 1:],
        dst_origin_imgs_sizes=data['original_hw'][:, 1:],
        depth_consistency_thres=0.005,
        cycle_reproj_distance_thres=1
    )  # valid_mask: [B, n_dst, n_pts], warpped_pts: [B, n_dst, n_pts, 2], world_pts: [B, n_pts, 3], scales_absolute: [B, n_view, n_pts], view_point_vector: [B, n_view, n_pts, 3]

    valid_mask = valid_mask * pad_mask[:, None] if pad_mask is not None else valid_mask

    # Find valid GT tracks and sample(or pad)
    n_query_view, n_pts = valid_mask.shape[1:]
    # track_valid_mask = torch.sum(valid_mask, dim=1) >= (
    #     n_query_view - track_length_tolerance
    # )  # [B, 1, n_pts]
    track_valid_mask = valid_mask[:, 0]  # [B, n_pts]
    # GT:
    reference_points = warpped_pts

    world_points = world_points

    query_points = query_coords  # [B, n_selected, 2]  # [B, n_selected, 2]

    # query_points = (query_points / scale[:, 0]).round()
    # reference_points = (reference_points / scale[:, 1:]).round()
    trajectory=torch.cat((query_points.unsqueeze(1), reference_points), dim=1)
    # trajectory = remap_track(trajectory, H, W, data['original_hw']).round()
    trajectory = (trajectory / scale).round()
    # NOTE: all points are in original scale
    return (
        CoTrackerData(
            video=data["video"].squeeze(0),# [B, N, C, H ,W]
            # trajectory=reference_points, # [B, n_dst, n_selected, 2]
            visibility=torch.cat((track_valid_mask.unsqueeze(1), valid_mask), dim=1).squeeze(0), # [B, N, n_selected]
            valid=torch.ones_like(track_valid_mask).squeeze(0), # [B, n_dst, n_selected]
            seq_name=data["scene_name"], #scene name of megadepth
            query_points=trajectory[:,0].squeeze(0), # [B, n_selected, 2]
            trajectory=trajectory.squeeze(0), # [B, N, n_selected, 2]
            image_list=data["image_list"], # [B, N]
        ),
        True,
    )

def remap_track(track, H, W, ori_size):
    """
    track: [B, N, T, 2] - 原始坐标
    H, W: 原始图像尺寸
    H1, W1: 缩放后图像尺寸
    """
    import cv2
    B, N, T, _ = track.shape

    track_resized = np.zeros_like(track)
    for b in range(B):
        for n in range(N):
            for t in range(T):
                # 创建像素映射表
                H1, W1 = ori_size[b, n]

                map_x, map_y = np.meshgrid(np.linspace(0, W-1, W1), np.linspace(0, H-1, H1))

                map_x = map_x.astype(np.float32)
                map_y = map_y.astype(np.float32)
                x, y = track[b, n, t]
                
                new_x = cv2.remap(map_x, np.array([[x]], dtype=np.float32), np.array([[y]], dtype=np.float32), interpolation=cv2.INTER_LANCZOS4)
                new_y = cv2.remap(map_y, np.array([[x]], dtype=np.float32), np.array([[y]], dtype=np.float32), interpolation=cv2.INTER_LANCZOS4)
                track_resized[b, n, t] = [new_x[0,0], new_y[0,0]]
    return torch.tensor(track_resized, device=track.device)