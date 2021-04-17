import os
import json

import numpy as np
import cv2
import torch
import smplx
from utils.common import json2kps, itt, to_tanh
from utils.bbox import crop_resize_coords, get_ltrb_bbox, crop_resize_verts, crop_resize_coords, crop_resize_image
import pickle


class InferenceDataset():
    def __init__(self, samples_dir, image_size, v_inds, smplx_model_path):
        self.samples_dir = samples_dir
        self.frame_list = self.list_samples(self.samples_dir)


        self.image_size = image_size
        self.input_size = image_size // 2


        print('loader self.image_size', self.image_size)
        print('loader self.input_size', self.input_size)
        self.v_inds = v_inds

        self.smplx_model = smplx.body_models.SMPLX(smplx_model_path)

    @staticmethod
    def list_samples(samples_dir):
        files = os.listdir(samples_dir)
        frame_ids = [x.split('_')[0] for x in files]
        frame_ids = sorted(list(set(frame_ids)))

        return frame_ids

    def load_rgb(self, frame_id):
        rgb_path = os.path.join(self.samples_dir, f"{frame_id}_rgb.png")
        rgb = cv2.imread(rgb_path)[..., ::-1] / 255.
        return itt(rgb)

    def load_segm(self, frame_id):
        rgb_path = os.path.join(self.samples_dir, f"{frame_id}_segm.png")
        rgb = cv2.imread(rgb_path)[..., ::-1] / 255.
        return itt(rgb)

    def load_landmarks(self, frame_id):
        landmarks_path = os.path.join(self.samples_dir, f"{frame_id}_keypoints.json")
        with open(landmarks_path, 'r') as f:
            landmarks = json.load(f)
        landmarks = json2kps(landmarks)
        landmarks = torch.FloatTensor(landmarks).unsqueeze(0)
        return landmarks

    def load_smplx(self, frame_id):
        smplx_path = os.path.join(self.samples_dir, f"{frame_id}_smplx.pkl")
        with open(smplx_path, 'rb') as f:
            smpl_params = pickle.load(f)

        for k, v in smpl_params.items():
            smpl_params[k] = torch.FloatTensor(v)

        with torch.no_grad():
            smpl_output = self.smplx_model(**smpl_params)
        vertices = smpl_output.vertices.detach()
        vertices = vertices[:, self.v_inds]
        K = smpl_params['camera_intrinsics'].unsqueeze(0)
        vertices = torch.bmm(vertices, K.transpose(1, 2))
        smpl_params.pop('camera_intrinsics')

        # vertices = vertices[0].numpy()
        # K = K[0].numpy()

        return vertices, K, smpl_params

    def __getitem__(self, item):
        frame_id = self.frame_list[item]

        rgb_orig = self.load_rgb(frame_id)
        segm_orig = self.load_segm(frame_id)
        landmarks_orig = self.load_landmarks(frame_id)
        verts_orig, K_orig, smpl_params = self.load_smplx(frame_id)

        ltrb = get_ltrb_bbox(verts_orig).float()
        vertices_crop, K_crop = crop_resize_verts(verts_orig, K_orig, ltrb, self.input_size)

        landmarks_crop = crop_resize_coords(landmarks_orig, ltrb, self.image_size)[0]
        rgb_crop = crop_resize_image(rgb_orig.unsqueeze(0), ltrb, self.image_size)[0]
        segm_crop = crop_resize_image(segm_orig.unsqueeze(0), ltrb, self.image_size)[0]

        rgb_crop = rgb_crop * segm_crop[:1]
        rgb_crop = to_tanh(rgb_crop)

        vertices_crop = vertices_crop[0]
        K_crop = K_crop[0]

        smpl_params = {k:v[0] for (k,v) in smpl_params.items()}
        data_dict = dict(real_rgb=rgb_crop, real_segm=segm_crop, landmarks=landmarks_crop, verts=vertices_crop,
                         K=K_crop)
        data_dict.update(smpl_params)

        return data_dict

    def __len__(self):
        return len(self.frame_list)




# def load_avakhitov_fits_vposer(vposer, part_path):
#     poses = np.load(part_path + '/poses.npy')[:-1]
#     face_expressions = np.load(part_path + '/expressions.npy')[:-1] * 1e2
#     betas = np.load(part_path + '/betas.npy')

#     with open(part_path + '/config.json', 'r') as f:
#         config = json.load(f)
        
#     # whether to use vposer embeddings
#     is_vposer = config['is_vposer']
#     is_male = config['is_male']

#     n = len(poses)

#     rot = poses[:, -3:]
#     trans = poses[:, -6:-3]

#     if is_vposer:
#         pose_body_vp = torch.tensor(poses[:, 0:32])
#         # convert from vposer to rotation matrices
#         pose_body_list = []
#         for i in range(n):
#             pose_body_mats = vposer.decode(pose_body_vp[i]).reshape(-1, 3, 3).detach().cpu().numpy()
#             pose_body = np.zeros(63)
#             for i in range(0, pose_body_mats.shape[0]):
#                 rot_vec, jac = cv2.Rodrigues(pose_body_mats[i])
#                 pose_body[3 * i: 3 * i + 3] = rot_vec.reshape(-1)
#             pose_body_list.append(pose_body)
#         pose_body = np.array(pose_body_list)
#         pose_jaw = poses[:, 32:35]
#         pose_eye = poses[:, 35:41]
#         pose_hand = poses[:, 41:-6]
#     else:
#         pose_body = poses[:, 0:63]
#         pose_jaw = poses[:, 63:66]
#         pose_eye = poses[:, 66:72]
#         pose_hand = poses[:, 72:-6]

#     result = {
#         'global_rvec': rot,
#         'global_tvec': trans,
#         'body_pose': pose_body,
#         'hand_pose': pose_hand,
#         'jaw_pose': pose_jaw,
#         'eye_pose': pose_eye,
#         'face_expression': face_expressions,
#         'betas': betas,
#         'n': n,
#         'is_male': is_male,
#         'is_vposer': is_vposer
#     }
#     return result


# class ExampleDataset:
#     def __init__(
#         self,
#         root_dir,
#         image_h=512,
#         image_w=512,
#         render_h=256,
#         render_w=256,
#         bbox_scale=1.2,
#         v_inds_path="./data/v_inds.npy",
#         vposer_dir="./data/vposer_v1_0"
#     ):

#         self.root_dir = root_dir
#         self.image_h, self.image_w = image_h, image_w
#         self.render_h, self.render_w = render_h, render_w
#         self.bbox_scale = bbox_scale

#         self.v_inds = np.load(v_inds_path)

#         # setup
#         self.smplx_dir = os.path.join(self.root_dir, "smplx")
#         self.image_dir = os.path.join(self.root_dir, "imgs")
#         self.openpose_dir = os.path.join(self.root_dir, "openpose")
#         self.smplx_params_dir = os.path.join(self.root_dir, "smplx_params")

#         self.mask_dir = os.path.join(self.root_dir, "masks")
#         if not os.path.exists(self.mask_dir):
#             self.mask_dir = os.path.join(self.root_dir, "segm")

#         # n frames total
#         self.n_frames = min(len(os.listdir(self.image_dir)), len(os.listdir(self.mask_dir)), len(glob.glob(os.path.join(self.smplx_dir, "*.npy"))))

#         # orig image shape
#         self.orig_h, self.orig_w = cv2.imread(os.path.join(self.image_dir, "{:06d}.png".format(0))).shape[:2]
        
#         # smplx camera params
#         with open(os.path.join(self.root_dir, "smplx_params", "calib.json")) as f:
#             self.K_orig = np.array(json.load(f)['0'])

#         # load smplx_params
#         vposer, _ = load_vposer(vposer_dir, vp_model='snapshot')
#         vposer.eval()

#         self.smplx_params = load_avakhitov_fits_vposer(vposer, self.smplx_params_dir)

#     def load_landmarks(self, path):
#         with open(path, 'r') as f:
#             landmarks = json.load(f)
#         landmarks = json2kps(landmarks)
#         return landmarks

#     def load_single_frame(self, frame_index):
#         # landmarks
#         openpose_path = os.path.join(self.openpose_dir, "{:06d}_keypoints.json".format(frame_index))

#         # smplx
#         smplx_path = os.path.join(self.smplx_dir, "{:06d}_xyz.npy".format(frame_index))

#         verts = np.load(smplx_path)

#         verts = verts @ self.K_orig.T
#         verts = verts[self.v_inds]

#         # crop
#         bbox = utils.bbox.get_bbox_from_smplifyx(verts)
#         bbox_h, bbox_w = bbox[2] - bbox[0], bbox[3] - bbox[1]

#         if bbox_h <= 0 or bbox_h > self.orig_h or bbox_w <= 0 or bbox_w > self.orig_w:
#             bbox = (0, 0, self.orig_h, self.orig_w)
#         l, t, r, b = bbox
#         ltrb = torch.tensor([l, t, r, b]).unsqueeze(dim=0)
#         ltrb = utils.bbox.get_square_bbox(ltrb)
#         ltrb = utils.bbox.scale_bbox(ltrb, self.bbox_scale)
#         bbox = ltrb.squeeze().detach().cpu().numpy()
        
#         image_shape = (bbox[2] - bbox[0], bbox[3] - bbox[1])
#         new_image_shape = (self.image_h, self.image_w)

#         verts, K_updated = utils.bbox.update_smplifyx_after_crop_and_resize(
#             verts, self.K_orig, bbox, image_shape, (self.render_h, self.render_w)
#         )
        
#         verts = torch.from_numpy(verts).type(torch.float32).unsqueeze(0)
        
#         # image
#         image_path = os.path.join(self.image_dir, "{:06d}.png".format(frame_index))
#         image = cv2.imread(image_path)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         image = utils.bbox.crop_image(image, bbox)
#         image = cv2.resize(image, new_image_shape)
        
#         image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)

#         # mask
#         mask_path = os.path.join(self.mask_dir, "{:06d}.png".format(frame_index))
#         mask = cv2.imread(mask_path)
#         mask = utils.bbox.crop_image(mask, bbox)
#         mask = np.max(mask, axis=-1)
#         mask = cv2.resize(mask, new_image_shape)
#         mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)

#         # landmarks
#         landmarks = self.load_landmarks(openpose_path)
#         landmarks = crop_resize_coords(landmarks, bbox, new_image_shape)
#         landmarks = torch.from_numpy(landmarks).type(torch.float32).unsqueeze(0)

#         # smplx_params
#         with open(os.path.join(self.smplx_params_dir, "config.json")) as f:
#             smplx_params_config = json.load(f)
            
#         with open(os.path.join(self.smplx_params_dir, "calib.json")) as f:
#             calib = json.load(f)
            
#         beta = np.load(os.path.join(self.smplx_params_dir, "betas.npy"))
#         expression = np.load(os.path.join(self.smplx_params_dir, "expressions.npy"))
#         pose = np.load(os.path.join(self.smplx_params_dir, "poses.npy"))

#         smplx_params = {
#             'is_male': smplx_params_config['is_male'],
#             'calib': calib[str(frame_index)],
#             'beta': beta,
#             'expression': expression[frame_index],
#             'pose': pose[frame_index],
#         }

#         sample = {
#             'image': image,
#             'mask': mask,
#             'verts': verts,
#             'bbox': bbox,
#             'K': K_updated,
#             'landmarks': landmarks,

#             'global_rvec': self.smplx_params['global_rvec'][frame_index],
#             'global_tvec': self.smplx_params['global_tvec'][frame_index],
#             'body_pose': self.smplx_params['body_pose'][frame_index],
#             'hand_pose': self.smplx_params['hand_pose'][frame_index],
#             'jaw_pose': self.smplx_params['jaw_pose'][frame_index],
#             'eye_pose': self.smplx_params['eye_pose'][frame_index],
#             'face_expression': self.smplx_params['face_expression'][frame_index],
#             'betas': self.smplx_params['betas'],
#             'is_male': self.smplx_params['is_male']
#         }
        
#         return sample


#     def load(self, frame_indices):
#         # if frame_indices is int – choose frames uniformly 
#         if isinstance(frame_indices, int):
#             frame_indices = list(range(self.n_frames))[::self.n_frames // frame_indices + 1]
        
#         samples = [self.load_single_frame(frame_index) for frame_index in frame_indices]

#         # pack
#         result = dict()

#         images = 2 * ((torch.cat([sample['image'] for sample in samples], dim=0) / 255.0) - 0.5)

#         masks = (torch.cat([sample['mask'] for sample in samples], dim=0) / 255.0)
#         masks = utils.common.segm2mask(masks)
#         images = utils.common.segment_img(images, masks)

#         result['image'] = images
#         result['segm'] = masks
        
#         result['verts'] = torch.cat([sample['verts'] for sample in samples], dim=0)

#         result['smplx'] = [{'verts': sample['verts'], 'calibration_matrix': sample['K']} for sample in samples]
#         result['calibration_matrix'] = np.stack([sample['K'] for sample in samples], axis=0)

#         result['landmarks'] = torch.cat([sample['landmarks'] for sample in samples], dim=0)

#         result['bbox'] = [sample['bbox'] for sample in samples]

#         # smplx_params
#         for key in ['global_rvec', 'global_tvec', 'body_pose', 'hand_pose', 'jaw_pose', 'eye_pose', 'face_expression', 'betas']:
#             result[key] = np.stack([sample[key] for sample in samples], axis=0)

#         result['gender'] = ['male' if sample['is_male'] else 'female' for sample in samples]

#         return result
