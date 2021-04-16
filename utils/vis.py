import os
import re
import json
import subprocess

import numpy as np
import torch
import cv2
from PIL import Image

import utils


def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().detach().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray


def normalize_image(image):
    """Normalizes image from [0.0, 255.0] range to [-1.0, 1.0]"""
    return 2 * (image / 255.0 - 0.5) 


def denormalize_image(image):
    """Denormalizes image from [-1.0, 1.0] to [0.0, 255.0]"""
    return 255.0 * (image + 1.0) / 2.0


def paste_image_on_canvas(image, canvas, bbox):
    bbox_h, bbox_w = bbox[2] - bbox[0], bbox[3] - bbox[1]

    canvas_pil = Image.fromarray(canvas.copy().astype(np.uint8))

    image_pil = Image.fromarray(image.copy().astype(np.uint8))
    image_pil = image_pil.resize((int(bbox_h), int(bbox_w)))

    canvas_pil.paste(image_pil, (int(bbox[0]), int(bbox[1])))

    return np.asarray(canvas_pil)


def image_batch_to_numpy(image_batch):
    image_batch = to_numpy(image_batch)
    image_batch = np.transpose(image_batch, (0, 2, 3, 1)) # BxCxHxW -> BxHxWxC
    return image_batch


def image_batch_to_torch(image_batch):
    image_batch = np.transpose(image_batch, (0, 3, 1, 2)) # BxHxWxC -> BxCxHxW
    image_batch = to_torch(image_batch).float()
    return image_batch


def vis_images_sbs(images_input, images_pred, n_samples=-1):
    batch_size = images_input.shape[0]
    n_samples = min(n_samples, batch_size) if n_samples > 0 else batch_size
    
    images_input = denormalize_image(image_batch_to_numpy(images_input)).astype(np.uint8)
    images_pred = denormalize_image(image_batch_to_numpy(images_pred)).astype(np.uint8)
    
    canvases = []
    for i in range(n_samples):
        canvases.append(np.concatenate([images_input[i], images_pred[i]], axis=1))
    
    return canvases


def vis_images(images, segms, white=True, n_samples=-1):
    batch_size = images.shape[0]
    n_samples = min(n_samples, batch_size) if n_samples > 0 else batch_size

    if white:
        segms = segms.repeat((1, 3, 1, 1))
        images = segms * images + (1.0 - segms) * torch.ones_like(images)
    
    images = denormalize_image(image_batch_to_numpy(images)).astype(np.uint8)
    
    canvases = images
    return canvases


def vis_uvs(uvs, masks, n_samples=-1):
    batch_size = uvs.shape[0]
    n_samples = min(n_samples, batch_size) if n_samples > 0 else batch_size

    uvs = (
        uvs[:n_samples]
        .permute(0, 2, 3, 1)
        .add(1.0)
        .div(2.0)
        .mul(255)
        .detach().cpu().numpy()
        .astype(np.uint8)
    )
    
    if masks is not None:
        masks = (
            masks[:n_samples]
            .permute(0, 2, 3, 1)
            .detach().cpu().numpy()
            .astype(np.uint8)
        )
        
        uvs = uvs * masks

    uvs = np.concatenate([uvs, np.zeros((*uvs.shape[:3], 1), dtype=np.uint8)], axis=-1)

    canvases = uvs
    return canvases


def vis_rotation(generator_inferer, latents, noise, verts, ntexture, calibration_matrices, white=True, n_rotation_samples=16, n_samples=-1):
    batch_size = verts.shape[0]
    n_samples = min(n_samples, batch_size) if n_samples > 0 else batch_size

    with torch.no_grad():    
        # generate ntexture from latents
        with torch.no_grad():
            infer_output_dict = generator_inferer.infer_pass(
                latents,
                verts,
                noises=noise,
                ntexture=ntexture,
                uv=None,
                return_latents=False,
                input_is_latent=True,
                infer_g_ema=True
            )

        ntexture = infer_output_dict['fake_ntexture']
        
        # generate rotation images from ntexture
        all_rotation_images = []

        for batch_i in range(n_samples):
            current_verts = verts[batch_i]

            current_calibration_matrix = calibration_matrices[batch_i]
            current_calibration_matrix_inv = torch.inverse(calibration_matrices[batch_i])

            # get uvs with rotation
            rotation_uvs = []
            for angle in np.linspace(0.0, 2 * np.pi, num=n_rotation_samples):
                rotation_matrix = utils.common.get_rotation_matrix(angle, axis='y')
                rotation_matrix = torch.from_numpy(rotation_matrix).type(torch.float32).to(verts.device)

                # denormalize
                current_verts_rotated = current_verts @ current_calibration_matrix_inv.t()

                # rotate
                mean_point = torch.mean(current_verts_rotated, dim=0)
                current_verts_rotated -= mean_point
                current_verts_rotated = current_verts_rotated @ rotation_matrix.t()
                current_verts_rotated += mean_point

                # normalize
                current_verts_rotated = current_verts_rotated @ current_calibration_matrix.t()

                # render uv
                uv = generator_inferer.uv_renderer.forward(current_verts_rotated.unsqueeze(0))
                rotation_uvs.append(uv)

            rotation_uvs = torch.cat(rotation_uvs, dim=0)

            # generated images
            infer_output_dict = generator_inferer.infer_pass(
                None,
                None,
                noises=noise,
                ntexture=ntexture[batch_i].unsqueeze(0).repeat(n_rotation_samples, 1, 1, 1),
                uv=rotation_uvs,
                return_latents=False,
                input_is_latent=True,
                infer_g_ema=True
            )
            rotation_images = infer_output_dict['fake_img']
            rotation_segms = infer_output_dict['fake_segm']

            if white:
                rotation_segms = rotation_segms.repeat((1, 3, 1, 1))
                rotation_images = rotation_segms * rotation_images + (1.0 - rotation_segms) * torch.ones_like(rotation_images)

            rotation_images = denormalize_image(image_batch_to_numpy(rotation_images)).astype(np.uint8)

            all_rotation_images.append(rotation_images)
            
    canvases = []
    for rotation_sample_i in range(n_rotation_samples):
        canvas = np.concatenate(
            [all_rotation_images[batch_i][rotation_sample_i] for batch_i in range(n_samples)],
            axis=0
        )
        canvases.append(canvas)
        
    return canvases


def vis_images_over_time(image_dir, contains):
    all_image_names = os.listdir(image_dir)

    image_names = []
    steps = []
    for image_name in all_image_names:
        # match = re.findall(fr"^{startswith}_(\d+)_.*", image_name)
        match = re.findall(fr".*{contains}_(\d+)_.*", image_name)
        if len(match) > 0:
            image_names.append(image_name)
            steps.append(int(match[0]))
            
    index_sorted = np.argsort(steps)
    image_names = [image_names[index] for index in index_sorted]
    steps = [steps[index] for index in index_sorted]

    image_names.sort(key=lambda x: os.stat(os.path.join(image_dir, x)).st_mtime)
    images = [cv2.cvtColor(cv2.imread(os.path.join(image_dir, image_name)), cv2.COLOR_BGR2RGB) for image_name in image_names]

    images_with_text = []
    for image, step in zip(images, steps):
        image_with_text = cv2.putText(
            image.copy(),
            str(step), 
            (10, 20), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5,
            (255, 255, 255),
            2
        )
        
        images_with_text.append(image_with_text)

    images_with_text = np.array(images_with_text)

    return images_with_text


def vis_seq(dataset, generator_inferer, latents, noise, ntexture, save_dir, white=False, device='cuda:0'):
    frame_dir = os.path.join(save_dir, "frames")
    os.makedirs(frame_dir, exist_ok=True)

    bbox_list = []
    for frame_index in range(dataset.n_frames):
        data_dict = dict()
        data_dict['image'], data_dict['mask'], data_dict['verts'], bbox, _, _ = dataset.load_single_frame(frame_index)
        data_dict = utils.common.dict2device(data_dict, device)

        bbox_list.append(bbox.tolist())

        # infer
        with torch.no_grad():
            infer_output_dict = generator_inferer.infer_pass(
                None if latents is None else torch.cat([latents] * len(data_dict['image']), dim=0),
                data_dict['verts'],
                noises=noise,
                ntexture=None if ntexture is None else torch.cat([ntexture] * len(data_dict['image']), dim=0),
                uv=None,
                return_latents=False,
                input_is_latent=False,
                infer_g_ema=True
            )

        image_pred = infer_output_dict['fake_img']
        segm_pred = infer_output_dict['fake_segm']

        if white:
            segm_pred = segm_pred.repeat((1, 3, 1, 1))
            image_pred = segm_pred * image_pred + (1.0 - segm_pred) * torch.ones_like(image_pred)

        image = denormalize_image(image_batch_to_numpy(image_pred)).astype(np.uint8)[0]
        frame_path = os.path.join(frame_dir, f"{frame_index:06d}.png")

        if white:
            canvas = 255 * np.ones((dataset.orig_h, dataset.orig_w, 3)).astype(np.uint8)
        else:
            canvas = np.zeros((dataset.orig_h, dataset.orig_w, 3)).astype(np.uint8)
        image = paste_image_on_canvas(image, canvas, bbox)

        cv2.imwrite(frame_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    with open(os.path.join(save_dir, "bboxes.json"), 'w') as f:
        json.dump(bbox_list, f)

    # create video
    output_video_path = os.path.join(os.path.join(save_dir, "video.mp4"))

    cmd = [
        "ffmpeg",
        "-y",
        "-framerate", "25",
        "-i", os.path.join(frame_dir, "%06d.png"),
        "-c:v", "libx264",
        "-vf", "fps=25,pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-pix_fmt", "yuv420p",
        output_video_path
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        raise ValueError(result.stderr.decode("utf-8"))

    return output_video_path