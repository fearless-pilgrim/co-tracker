import io
import PIL
import cv2
import h5py
import torch
import numpy as np
from PIL import Image
from loguru import logger

def process_resize(w, h, resize, df=None, resize_no_larger_than=False):
    assert(len(resize) > 0 and len(resize) <= 2)
    if resize_no_larger_than and (max(h, w) <= max(resize)):
        w_new, h_new = w, h
    else:
        if len(resize) == 1 and resize[0] > -1:  # resize the larger side
            scale = resize[0] / max(h, w)
            w_new, h_new = int(round(w*scale)), int(round(h*scale))
        elif len(resize) == 1 and resize[0] == -1:
            w_new, h_new = w, h
        else:  # len(resize) == 2:
            w_new, h_new = resize[1], resize[0]

    if df is not None:
        w_new, h_new = map(lambda x: int(x // df * df), [w_new, h_new])
    return w_new, h_new

def resize_image(image, size, interp):
    # NOTE: from hloc
    if interp.startswith('cv2_'):
        interp = getattr(cv2, 'INTER_'+interp[len('cv2_'):].upper())
        h, w = image.shape[:2]
        if interp == cv2.INTER_AREA and (w < size[0] or h < size[1]):
            interp = cv2.INTER_LINEAR
        resized = cv2.resize(image, size, interpolation=interp)
    elif interp.startswith('pil_'):
        interp = getattr(PIL.Image, interp[len('pil_'):].upper())
        resized = PIL.Image.fromarray(image.astype(np.uint8))
   
        resized = resized.resize(size, resample=interp)
  
        resized = np.asarray(resized, dtype=image.dtype)
  
    else:
        raise ValueError(
            f'Unknown interpolation {interp}.')
    return resized

def pad_bottom_right(inp, pad_size, ret_mask=False):
    
    assert isinstance(pad_size, int) and pad_size >= max(inp.shape[-2:]), f"{pad_size} < {max(inp.shape[-2:])}"
    mask = None
    if inp.ndim == 2:
        padded = np.zeros((pad_size, pad_size), dtype=inp.dtype)
        padded[:inp.shape[0], :inp.shape[1]] = inp
        if ret_mask:
            mask = np.zeros((pad_size, pad_size), dtype=bool)
            mask[:inp.shape[0], :inp.shape[1]] = True
    elif inp.ndim == 3:
        padded = np.zeros((pad_size, pad_size, inp.shape[-1]), dtype=inp.dtype)
        padded[:inp.shape[0], :inp.shape[1], :] = inp
        if ret_mask:
            mask = np.zeros((pad_size, pad_size, inp.shape[-1]), dtype=bool)
            mask[:inp.shape[0], :inp.shape[1], :] = True
    else:
        raise NotImplementedError()
    return padded, mask

def read_megadepth_depth(path, resize=None, client=None, pad_to=None):
    depth = np.array(h5py.File(path, 'r')['/depth']) if client is None \
        else load_array_from_petrel(path, client, None, use_h5py=True)  # (h, w)


    if pad_to is not None:
        depth, _ = pad_bottom_right(depth, pad_to)

    if resize is not None:
        return resize_and_pad_depth(depth, (resize[1], resize[0]), pad_to)

    return torch.from_numpy(depth).float() 

def adjust_intrinsic(K, scales, padding=False, paddings=None):
    """
    调整相机内参 K 以适应 resize 和 padding 操作。

    :param K: 原始 3x3 相机内参矩阵
    :param scale_w: 宽度方向的缩放因子
    :param scale_h: 高度方向的缩放因子
    :param pad_w: 左右方向 padding 的像素数
    :param pad_h: 上下方向 padding 的像素数
    :return: 调整后的 3x3 相机内参矩阵
    """
    K_new = K.copy()
    scale_h, scale_w = scales
    K_new[0] /= scale_w  # 调整 fx
    K_new[1] /= scale_h  # 调整 fy
    if padding:
        pad_w, pad_h = paddings
        K_new[0, 2] = K_new[0, 2] * scale_w + pad_w  # 调整 cx
        K_new[1, 2] = K_new[1, 2] * scale_h + pad_h  # 调整 cy
    return K_new

def resize_and_pad_depth(depth, target_size, pad_size, device='cuda'):
    """
    对深度图进行 resize 和 padding，使其与 RGB 图像对齐
    :param depth: numpy array, 原始深度图 (H, W)
    :param target_size: tuple (new_W, new_H), 目标大小
    :param pad_size: tuple (pad_W, pad_H), 需要填充到的最终大小
    :param device: 目标设备
    :return: 处理后的深度图 (torch.Tensor)
    """

    # Step 1: Resize（使用最近邻插值，防止深度值混合）
    resized_depth = Image.fromarray(depth)  # 转换为PIL对象
    resized_depth = resized_depth.resize(target_size, resample=Image.NEAREST)  # 最近邻插值
    resized_depth = np.array(resized_depth)  # 转换回 NumPy

    # Step 2: Padding（填充右下角）
    if pad_size is None:
        return torch.from_numpy(resized_depth).float().to(device)
    pad_w, pad_h = pad_size
    h, w = resized_depth.shape
    padded_depth = np.full((pad_h, pad_w), fill_value=0, dtype=np.float32)  # 以0填充
    padded_depth[:h, :w] = resized_depth  # 将resize后的深度图放置到左上角

    # 转换为 PyTorch Tensor 并移动到 GPU
    padded_depth = torch.from_numpy(padded_depth).float().to(device)

    return padded_depth

def read_rgb(path, resize=None, resize_no_larger_than=False, resize_float=False, df=None, client=None,
                   pad_to=None, ret_scales=False, ret_pad_mask=False,
                   augmentor=None):
    
    resize = tuple(resize) if resize is not None else None
    if augmentor is None:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR) if client is None \
            else load_array_from_petrel(path, client, cv2.IMREAD_GRAYSCALE)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR) if client is None \
            else load_array_from_petrel(path, client, cv2.IMREAD_COLOR)  # BGR image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = augmentor(image)

    if image is None:
        logger.error(f"Problem exists when loading image: {path}")
    
    # import ipdb; ipdb.set_trace()
    w, h = image.shape[1], image.shape[0]
    w_new, h_new = process_resize(w, h, resize if resize is not None else (w, h), df, resize_no_larger_than=resize_no_larger_than)
    scales = torch.tensor([float(h) / float(h_new), float(w) / float(w_new)]) # [2]
    # original_hw = torch.tensor([w_new, h_new]) #[2]
    original_hw = torch.tensor([h, w])
    
    image = resize_image(image, (w_new, h_new), interp="pil_LANCZOS").astype('float32')

    if pad_to is not None:
        if pad_to == -1:
            pad_to = max(w_new, h_new)

        image, mask = pad_bottom_right(image, pad_to, ret_mask=ret_pad_mask)

    # print(f"==> resize image shape from {w, h} to {image.shape}")
    ts_image = rgb2tensor(image)
    ret_val = [ts_image]
    ret_val += [scales, original_hw]
    if ret_scales:
        ret_val += [scales, original_hw]
    if ret_pad_mask:
        ts_mask = mask2tensor(mask) if pad_to else None
        ret_val.append(ts_mask if pad_to else None)
        
    return ret_val[0] if len(ret_val) == 1 else ret_val

def rgb2tensor(image):
    # return torch.from_numpy(image/255.).float().permute(2, 0, 1).contiguous()  # (3, h, w)
    return torch.from_numpy(image).float().permute(2, 0, 1).contiguous()  # (3, h, w)

def mask2tensor(mask):
    return torch.from_numpy(mask).float()  # (h, w)

def load_array_from_petrel(path, client, cv_type, max_retry=10, use_h5py=False, return_tensor=True):
    byte_str = client.Get(path)
    try:
        if not use_h5py:
            raw_array = np.fromstring(byte_str, np.uint8)
            data = cv2.imdecode(raw_array, cv_type)
        else:
            f = io.BytesIO(byte_str)
            data = np.array(h5py.File(f, 'r')['/depth'])
    except Exception as ex:
        print(f"==> Data loading failure: {path}")
        raise ex

    assert data is not None
    return data