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


def compute_bboxes_from_keypoints(keypoints):
    """
    keypoints: B x 68*2
    return value: B x 4 (t, b, l, r)
    Compute a very rough bounding box approximate from 68 keypoints.
    """
    x, y = keypoints.float().view(-1, 68, 2).transpose(0, 2)

    face_height = y[8] - y[27]
    b = y[8] + face_height * 0.2
    t = y[27] - face_height * 0.47

    midpoint_x = (x.min(dim=0)[0] + x.max(dim=0)[0]) / 2
    half_height = (b - t) * 0.5

    l = midpoint_x - half_height
    r = midpoint_x + half_height


    return torch.stack([t, b, l, r], dim=1)


# def crop_and_resize(images, bboxes, target_size=None):
#     """
#     images: B x C x H x W
#     bboxes: B x 4; [t, b, l, r], in pixel coordinates
#     target_size (optional): tuple (h, w)

#     return value: B x C x h x w

#     Crop i-th image using i-th bounding box, then resize all crops to the
#     desired shape (default is the original images' size, H x W).
#     """

#     t, b, l, r = bboxes.t().float()
#     batch_size, num_channels, h, w = images.shape

#     affine_matrix = torch.zeros(batch_size, 2, 3, dtype=torch.float32, device=images.device)
#     affine_matrix[:, 0, 0] = (r-l) / w
#     affine_matrix[:, 1, 1] = (b-t) / h
#     affine_matrix[:, 0, 2] = (l+r) / w - 1
#     affine_matrix[:, 1, 2] = (t+b) / h - 1

#     output_shape = (batch_size, num_channels) + (target_size or (h, w))
#     grid = torch.affine_grid_generator(affine_matrix, output_shape, align_corners=True)
#     grid = grid.to(images.dtype)
#     return torch.nn.functional.grid_sample(images, grid, 'bilinear', 'reflection', align_corners=True)


# import numpy as np
# import torch

# from PIL import Image


# def get_ltrb_from_verts(verts):
#     verts_projected = (verts / (verts[..., 2:]))[..., :2]

#     x = verts_projected[..., 0]
#     y = verts_projected[..., 1]

#     # get bbox in format (left, top, right, bottom)
#     l = torch.min(x, dim=1)[0].long()
#     t = torch.min(y, dim=1)[0].long()
#     r = torch.max(x, dim=1)[0].long()
#     b = torch.max(y, dim=1)[0].long()

#     return torch.stack([l, t, r, b], dim=-1)


# def get_square_bbox(ltrb):
#     """Makes square bbox from any bbox by stretching of minimal length side

#     Args:
#         bbox tuple of size 4: input bbox (left, upper, right, lower)

#     Returns:
#         bbox: tuple of size 4:  resulting square bbox (left, upper, right, lower)
#     """

#     left = ltrb[:, 0]
#     right = ltrb[:, 2]
#     top = ltrb[:, 1]
#     bottom = ltrb[:, 3]

#     width, height = right - left, bottom - top

#     if width > height:
#         y_center = (ltrb[:, 3] + ltrb[:, 1]) // 2
#         top = y_center - width // 2
#         bottom = top + width
#     else:
#         x_center = (ltrb[:, 2] + ltrb[:, 0]) // 2
#         left = x_center - height // 2
#         right = left + height

#     new_ltrb = torch.stack([left, top, right, bottom], dim=-1)
#     return new_ltrb


# def scale_bbox(ltrb, scale):
#     width, height = ltrb[:, 2] - ltrb[:, 0], ltrb[:, 3] - ltrb[:, 1]

#     x_center, y_center = (ltrb[:, 2] + ltrb[:, 0]) // 2, (ltrb[:, 3] + ltrb[:, 1]) // 2
#     new_width, new_height = int(scale * width), int(scale * height)

#     new_left = x_center - new_width // 2
#     new_right = new_left + new_width

#     new_top = y_center - new_height // 2
#     new_bottom = new_top + new_height

#     new_ltrb = torch.stack([new_left, new_top, new_right, new_bottom], dim=-1)

#     return new_ltrb


# def get_ltrb_bbox(verts, scale=1.2):
#     ltrb = get_ltrb_from_verts(verts)

#     ltrb = scale_bbox(ltrb, scale)
#     ltrb = get_square_bbox(ltrb)

#     return ltrb


# def compute_bboxes_from_keypoints(keypoints):
#     """
#     keypoints: B x 68*2

#     return value: B x 4 (t, b, l, r)

#     Compute a very rough bounding box approximate from 68 keypoints.
#     """
#     x, y = keypoints.float().view(-1, 68, 2).transpose(0, 2)

#     face_height = y[8] - y[27]
#     b = y[8] + face_height * 0.2
#     t = y[27] - face_height * 0.47

#     midpoint_x = (x.min(dim=0)[0] + x.max(dim=0)[0]) / 2
#     half_height = (b - t) * 0.5

#     l = midpoint_x - half_height
#     r = midpoint_x + half_height


#     return torch.stack([t, b, l, r], dim=1)


# def crop_image(image, bbox, return_mask=False):
#     """Crops area from image specified as bbox. Always returns area of size as bbox filling missing parts with zeros
#     Args:
#         image numpy array of shape (height, width, 3): input image
#         bbox tuple of size 4: input bbox (left, upper, right, lower)

#     Returns:
#         cropped_image numpy array of shape (height, width, 3): resulting cropped image

#     """
#     image_pil = Image.fromarray(image, mode='RGB')
#     image_pil = image_pil.crop(bbox)

#     if return_mask:
#         mask = np.ones_like(image)
#         mask_pil = Image.fromarray(mask, mode='RGB')
#         mask_pil = mask_pil.crop(bbox)
#         return np.asarray(image_pil), np.asarray(mask_pil)[..., :1]

#     return np.asarray(image_pil)


# def crop_resize_image(image, ltrb, new_image_size=None):
#     if new_image_size is None:
#         new_image_size = image.shape
#     size = new_image_size

#     l, t, r, b = ltrb.t().float()
#     batch_size, num_channels, h, w = image.shape

#     affine_matrix = torch.zeros(batch_size, 2, 3, dtype=torch.float32, device=image.device)
#     affine_matrix[:, 0, 0] = (r - l) / w
#     affine_matrix[:, 1, 1] = (b - t) / h
#     affine_matrix[:, 0, 2] = (l + r) / w - 1
#     affine_matrix[:, 1, 2] = (t + b) / h - 1

#     output_shape = (batch_size, num_channels) + (size, size)
#     grid = torch.affine_grid_generator(affine_matrix, output_shape, align_corners=True)
#     grid = grid.to(image.dtype)
#     return torch.nn.functional.grid_sample(image, grid, 'bilinear', 'reflection', align_corners=True)





# def crop_resize_coords(coords, ltrb, imsize):
#     invalid_mask = coords.sum(axis=-1) <= 0
#     crop_j, crop_i, r, b = ltrb
#     crop_sz = r - crop_j

#     resize_ratio = imsize / crop_sz
#     cropped = coords.copy()
#     cropped[:, 1] -= crop_i
#     cropped[:, 0] -= crop_j
#     cropped[:, :2] *= resize_ratio
#     cropped[invalid_mask] = -1
#     return cropped


# def crop_resize_verts(verts, K, ltrb, new_image_size):
#     # it's supposed that it smplifyx's verts are in trivial camera coordinates
#     fx, fy, cx, cy = 1.0, 1.0, 0.0, 0.0
#     # crop
#     cx, cy = cx - ltrb[:, 0], cy - ltrb[:, 1]
#     # scale
#     width, height = ltrb[:, 2] - ltrb[:, 0], ltrb[:, 3] - ltrb[:, 1]
#     new_h = new_w = new_image_size

#     h_scale, w_scale = new_w / width.float(), new_h / height.float()

#     fx, fy = fx * w_scale, fy * h_scale
#     cx, cy = cx * w_scale, cy * h_scale

#     # update verts
#     B = verts.shape[0]
#     K_upd = torch.eye(3)
#     K_upd = torch.stack([K_upd] * B, dim=0).to(verts.device)

#     K_upd[:, 0, 0] = fx
#     K_upd[:, 1, 1] = fy
#     K_upd[:, 0, 2] = cx
#     K_upd[:, 1, 2] = cy

#     verts_cropped = torch.bmm(verts, K_upd.transpose(1, 2))
#     K_cropped = torch.bmm(K_upd, K)

#     return verts_cropped, K_cropped


# def get_bbox_from_smplifyx(verts):
#     verts_projected = (verts / verts[:, 2:])[:, :2]

#     # get bbox in format (left, top, right, bottom)
#     l = np.min(verts_projected[:, 0])
#     t = np.min(verts_projected[:, 1])
#     r = np.max(verts_projected[:, 0])
#     b = np.max(verts_projected[:, 1])

#     return (l, t, r, b)


# def update_smplifyx_after_crop_and_resize(verts, K, bbox, image_shape, new_image_shape):
#     # it's supposed that it smplifyx's verts are in trivial camera coordinates
#     fx, fy, cx, cy = 1.0, 1.0, 0.0, 0.0

#     # crop
#     cx, cy = cx - bbox[0], cy - bbox[1]

#     # scale
#     h, w = image_shape
#     new_h, new_w = new_image_shape

#     h_scale, w_scale = new_w / w, new_h / h

#     fx, fy = fx * w_scale, fy * h_scale
#     cx, cy = cx * w_scale, cy * h_scale

#     # update verts
#     new_K = np.array([
#         [fx, 0.0, cx],
#         [0.0, fy, cy],
#         [0.0, 0.0, 1.0]
#     ])

#     return verts @ new_K.T, new_K @ K