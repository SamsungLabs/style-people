import numpy as np
import torch


def get_ltrb_from_verts(verts):
    verts_projected = (verts / (verts[..., 2:]))[..., :2]

    x = verts_projected[..., 0]
    y = verts_projected[..., 1]

    # get bbox in format (left, top, right, bottom)
    l = torch.min(x, dim=1)[0].long()
    t = torch.min(y, dim=1)[0].long()
    r = torch.max(x, dim=1)[0].long()
    b = torch.max(y, dim=1)[0].long()

    return torch.stack([l, t, r, b], dim=-1)


def scale_bbox(ltrb, scale):
    width, height = ltrb[:, 2] - ltrb[:, 0], ltrb[:, 3] - ltrb[:, 1]

    x_center, y_center = (ltrb[:, 2] + ltrb[:, 0]) // 2, (ltrb[:, 3] + ltrb[:, 1]) // 2
    new_width, new_height = int(scale * width), int(scale * height)

    new_left = x_center - new_width // 2
    new_right = new_left + new_width

    new_top = y_center - new_height // 2
    new_bottom = new_top + new_height

    new_ltrb = torch.stack([new_left, new_top, new_right, new_bottom], dim=-1)

    return new_ltrb


def get_square_bbox(ltrb):
    """Makes square bbox from any bbox by stretching of minimal length side

    Args:
        bbox tuple of size 4: input bbox (left, upper, right, lower)

    Returns:
        bbox: tuple of size 4:  resulting square bbox (left, upper, right, lower)
    """

    left = ltrb[:, 0]
    right = ltrb[:, 2]
    top = ltrb[:, 1]
    bottom = ltrb[:, 3]

    width, height = right - left, bottom - top

    if width > height:
        y_center = (ltrb[:, 3] + ltrb[:, 1]) // 2
        top = y_center - width // 2
        bottom = top + width
    else:
        x_center = (ltrb[:, 2] + ltrb[:, 0]) // 2
        left = x_center - height // 2
        right = left + height

    new_ltrb = torch.stack([left, top, right, bottom], dim=-1)
    return new_ltrb


def get_ltrb_bbox(verts, scale=1.2):
    ltrb = get_ltrb_from_verts(verts)

    ltrb = scale_bbox(ltrb, scale)
    ltrb = get_square_bbox(ltrb)

    return ltrb


def crop_resize_image(image, ltrb, new_image_size):
    size = new_image_size

    l, t, r, b = ltrb.t().float()
    batch_size, num_channels, h, w = image.shape

    affine_matrix = torch.zeros(batch_size, 2, 3, dtype=torch.float32, device=image.device)
    affine_matrix[:, 0, 0] = (r - l) / w
    affine_matrix[:, 1, 1] = (b - t) / h
    affine_matrix[:, 0, 2] = (l + r) / w - 1
    affine_matrix[:, 1, 2] = (t + b) / h - 1

    output_shape = (batch_size, num_channels) + (size, size)
    grid = torch.affine_grid_generator(affine_matrix, output_shape, align_corners=True)
    grid = grid.to(image.dtype)
    return torch.nn.functional.grid_sample(image, grid, 'bilinear', 'reflection', align_corners=True)


def crop_resize_coords(coords, ltrb, new_image_size):
    coords = coords.clone()

    width = ltrb[:, 2] - ltrb[:, 0]
    heigth = ltrb[:, 3] - ltrb[:, 1]

    coords[..., 0] -= ltrb[:, 0]
    coords[..., 1] -= ltrb[:, 1]

    coords[..., 0] *= new_image_size / width
    coords[..., 1] *= new_image_size / heigth

    return coords


def crop_resize_verts(verts, K, ltrb, new_image_size):
    # it's supposed that it smplifyx's verts are in trivial camera coordinates
    fx, fy, cx, cy = 1.0, 1.0, 0.0, 0.0
    # crop
    cx, cy = cx - ltrb[:, 0], cy - ltrb[:, 1]
    # scale
    width, height = ltrb[:, 2] - ltrb[:, 0], ltrb[:, 3] - ltrb[:, 1]
    new_h = new_w = new_image_size

    h_scale, w_scale = new_w / width.float(), new_h / height.float()

    fx, fy = fx * w_scale, fy * h_scale
    cx, cy = cx * w_scale, cy * h_scale

    # update verts
    B = verts.shape[0]
    K_upd = torch.eye(3)
    K_upd = torch.stack([K_upd] * B, dim=0).to(verts.device)

    K_upd[:, 0, 0] = fx
    K_upd[:, 1, 1] = fy
    K_upd[:, 0, 2] = cx
    K_upd[:, 1, 2] = cy

    verts_cropped = torch.bmm(verts, K_upd.transpose(1, 2))
    K_cropped = torch.bmm(K_upd, K)

    return verts_cropped, K_cropped
