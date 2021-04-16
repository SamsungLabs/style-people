import os
from tqdm import tqdm
import numpy as np
import pydoc
import random
import pickle
import copy
import socket
import cv2

import torch
from torch import nn

import lpips
import kornia

import utils
import inference_module
from utils.bbox import compute_bboxes_from_keypoints, crop_and_resize
from utils.vis import vis_images_sbs, vis_rotation

from moviepy.editor import ImageSequenceClip

from inference_module.criterion import LPIPSCriterion, MSECriterion, MAECriterion, SSIMCriterion, \
    UnaryDiscriminatorNonsaturatingCriterion, UnaryDiscriminatorFeatureMatchingCriterion
    

class Runner:
    def __init__(self, config, inferer, device='cuda:0'):
        self.config = config
        self.inferer = inferer

        # load texture segm
        with open(config.runner.texture_segm_path, 'rb') as f:
            texture_segm_dict = pickle.load(f)

        self.texture_segm_dict = utils.common.dict2device(texture_segm_dict, self.config.device, dtype=torch.float32)
        
        # load criterions
        ## lpips
        self.lpips_criterion = LPIPSCriterion(net='vgg').to(device)

        ## mse
        self.mse_criterion = MSECriterion().to(device)

        ## encoder_latent_deviation
        self.encoder_latent_deviation_criterion = MAECriterion().to(device)

        ## generator_params_deviation
        self.generator_params_deviation_criterion = MAECriterion().to(device)

        ## ntexture_deviation
        self.ntexture_deviation_criterion = MAECriterion().to(device)

        ## mae
        self.mae_criterion = MAECriterion().to(device)

        ## ssim
        self.ssim_criterion = SSIMCriterion(window_size=5, max_val=1, reduction='mean').to(device)

        ## face_lpips
        self.face_lpips_criterion = LPIPSCriterion(net='vgg').to(device)

        ## unary_face_discriminator_feature_matching
        self.unary_face_discriminator_feature_matching_criterion = UnaryDiscriminatorFeatureMatchingCriterion(
            self.inferer.discriminators['face'],
            endpoints=['reduction_' + str(i) for i in range(1,8)]
        ).to(device)

        self._init_state()
        

    def _init_state(self):
        self.state = {
            'epoch': 0,

            'train': {
                'n_batches_passed': 0,
                'n_samples_passed': 0
            },

            'val': {
                'n_batches_passed': 0,
                'n_samples_passed': 0
            }
        }

    def get_state_dict(self):
        state_dict = {
            'inferer': self.inferer.get_state_dict(),

            'meta': {
                'epoch': self.state['epoch'],
                'n_batches_passed': self.state['train']['n_batches_passed'],
                'n_samples_passed': self.state['train']['n_samples_passed']
            }
        }

        return state_dict

    def _get_latent_optimization_group(self, lr):
        optimization_group = {
            'params': [self.inferer.latent],
            'lr': lr
        }
        return optimization_group

    def _get_generator_optimization_group(self, lr):
        generator_params = []
        
        # convs
        generator_params.extend(self.inferer.g_ema.conv1.parameters())
        for l in self.inferer.g_ema.convs:
            generator_params.extend(l.parameters())
        
        # to_rgbs
        generator_params.extend(self.inferer.g_ema.to_rgb1.parameters())
        for l in self.inferer.g_ema.to_rgbs:
            generator_params.extend(l.parameters())

        optimization_group = {
            'params': generator_params,
            'lr': lr
        }

        return optimization_group

    def _get_noise_optimization_group(self, lr):
        optimization_group = {
            'params': self.inferer.noise,
            'lr': lr
        }
        return optimization_group

    def _get_ntexture_optimization_group(self, lr):
        optimization_group = {
            'params': self.inferer.ntexture,
            'lr': lr
        }
        return optimization_group

    def _get_beta_optimization_group(self, lr):
        optimization_group = {
            'params': self.inferer.betas,
            'lr': lr
        }
        return optimization_group

    def _get_hand_pose_optimization_group(self, lr):
        optimization_group = {
            'params': self.inferer.hand_pose,
            'lr': lr
        }
        return optimization_group

    def _get_body_pose_optimization_group(self, lr):
        optimization_group = {
            'params': self.inferer.body_pose,
            'lr': lr
        }
        return optimization_group

    def _get_optimization_group(self, name, lr):
        if name == 'latent':
            optimization_group = self._get_latent_optimization_group(lr)
        elif name == 'generator':
            optimization_group = self._get_generator_optimization_group(lr)
        elif name == 'noise':
            optimization_group = self._get_noise_optimization_group(lr)
        elif name == 'ntexture':
            optimization_group = self._get_ntexture_optimization_group(lr)
        elif name == 'beta':
            optimization_group = self._get_beta_optimization_group(lr)
        elif name == 'hand_pose':
            optimization_group = self._get_hand_pose_optimization_group(lr)
        elif name == 'body_pose':
            optimization_group = self._get_body_pose_optimization_group(lr)
        else:
            raise NotImplementedError(f"Unknown name {name}")

        return optimization_group


    def crop_face_with_landmarks(self, image, landmarks, face_size=(256, 256)):
        face_kp = landmarks[:, 25:93].clone()

        valid_mask = (face_kp[..., 0] < 0).sum(dim=1) <= 0
        invalid_mask = torch.logical_not(valid_mask)
        
        # add dummy valid landmarks for invalid images
        face_kp[invalid_mask] = torch.ones(
            torch.sum(invalid_mask), face_kp.shape[1], face_kp.shape[2], device=face_kp.device
        )
        
        bboxes_estimate = compute_bboxes_from_keypoints(face_kp)
        face = crop_and_resize(image, bboxes_estimate, face_size)
        
        return face, valid_mask

    def run_epoch(self, train_dict, val_dict, save_dir='.'):
        os.makedirs(os.path.join(save_dir, 'sbs'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'rotations'), exist_ok=True)
        
        self.inferer.eval()

        # get smplx betas, hand and body pose
        self.inferer.betas = train_dict['betas']
        self.inferer.betas.requires_grad_(True)

        self.inferer.hand_pose = train_dict['hand_pose']
        self.inferer.hand_pose.requires_grad_(True)

        self.inferer.body_pose = train_dict['body_pose']
        self.inferer.body_pose.requires_grad_(True)

        assert all(x == train_dict['gender'][0] for x in train_dict['gender'])
        gender = train_dict['gender'][0]

        v_inds = np.load(self.config.runner.v_inds_path)

        # save initial generator params
        generator_params_inital = utils.common.flatten_parameters(
            self.inferer.g_ema.parameters()
        ).detach().clone()
        
        # init with encoder
        if self.config.runner.init_with_encoder:
            encoder_latent = self.inferer.encoder.forward(kornia.resize(train_dict['image'], (self.config.encoder.image_size , self.config.encoder.image_size )))
            encoder_latent = encoder_latent.mean(0, keepdim=True)  # TODO: more smart averaging of latent vectors
            self.inferer.latent.data = encoder_latent.detach().clone()

        # save initial ntexture
        with torch.no_grad():
            infer_output_dict = self.inferer.infer_pass(
                torch.cat([self.inferer.latent] * len(train_dict['image']), dim=0),
                train_dict['verts'],
                noises=self.inferer.noise,
                ntexture=None,
                uv=None,
                return_latents=False,
                input_is_latent=True,
                infer_g_ema=True
            )
        initial_ntexture = infer_output_dict['fake_ntexture'][:1].detach().clone()

        batch_metrics = dict()
        stages = sorted(self.config.stages.keys())
        for stage in stages:
            stage_config = self.config.stages[stage]

            # maybe switch input source
            if stage_config.input_source == 'ntexture':
#                 print("Switching to ntexture")
                with torch.no_grad():
                    infer_output_dict = self.inferer.infer_pass(
                        torch.cat([self.inferer.latent] * len(train_dict['image']), dim=0),
                        train_dict['verts'],
                        noises=self.inferer.noise,
                        ntexture=None,
                        uv=None,
                        return_latents=False,
                        input_is_latent=True,
                        infer_g_ema=True
                    )

                    self.inferer.ntexture = nn.Parameter(infer_output_dict['fake_ntexture'][:1].detach().clone(), requires_grad=True)

                    self.inferer.latent = None
                    self.inferer.noise = None

                    # copy hands from initial texture
                    if self.config.runner.copy_hand_texture:
                        hand_texture_segm = self.texture_segm_dict['hands'].unsqueeze(0).unsqueeze(0)
                        self.inferer.ntexture = nn.Parameter(
                            hand_texture_segm * initial_ntexture + (1 - hand_texture_segm) * self.inferer.ntexture
                        )

            # setup optimizer
            optimization_groups = []
            for optimization_target in stage_config.optimization_targets:
                optimization_groups.append(
                    self._get_optimization_group(optimization_target, stage_config.lr[optimization_target])
                )
        
            optimizer = torch.optim.Adam(optimization_groups)

            # run optimization
            pbar = tqdm(range(stage_config.n_iters))
            pbar.set_description(f"{self.config.log.experiment_name}")
            for i in pbar:
                for mode, data_dict in [('val', val_dict), ('train', train_dict)]:
                    if mode =='val' and self.state['train']['n_batches_passed'] % self.config.log.log_val_freq != 0:
                        continue

                    torch.autograd.set_grad_enabled(mode == 'train')

                    # infer
                    smplx_output = self.inferer.smplx_models[gender](
                        global_orient=data_dict['global_rvec'],
                        transl=data_dict['global_tvec'],
                        betas=self.inferer.betas,
                        expression=data_dict['face_expression'],
                        body_pose=self.inferer.body_pose,
                        left_hand_pose=self.inferer.hand_pose[:, :12],
                        right_hand_pose=self.inferer.hand_pose[:, 12:],
                        jaw_pose=data_dict['jaw_pose'],
                        return_verts=True,
                        pose2rot=True,
                        # batch_size=1
                    )

                    verts_3d = smplx_output.vertices
                    verts_3d = verts_3d[:, v_inds]

                    verts_ndc, matrix_ndc = self.inferer.diff_uv_renderer.convert_to_ndc(
                        verts_3d,
                        data_dict['calibration_matrix'],
                        256, 256,
                        near=0.01, far=20.0,
                        invert_verts=False
                    )

                    uv, uv_da, mask, rast = self.inferer.diff_uv_renderer.render(verts_ndc, matrix_ndc, render_h=256, render_w=256)

                    infer_output_dict = self.inferer.infer_pass(
                        None if self.inferer.latent is None else torch.cat([self.inferer.latent] * len(data_dict['image']), dim=0),
                        None,
                        noises=self.inferer.noise,
                        ntexture=None if self.inferer.ntexture is None else torch.cat([self.inferer.ntexture] * len(data_dict['image']), dim=0),
                        uv=uv,
                        return_latents=False,
                        input_is_latent=True,
                        infer_g_ema=True
                    )

                    image_pred = infer_output_dict['fake_img']
                    segm_pred = infer_output_dict['fake_segm']
                    ntexture_pred = infer_output_dict['fake_ntexture']
                    
                    texture_out_path = os.path.join(save_dir, 'texture.pth')
                    torch.save(ntexture_pred.cpu(), texture_out_path)
                    
                    # calculate losses
                    loss = 0.0

                    ## lpips
                    if stage_config.loss_weight.lpips:
                        lpips_loss = self.lpips_criterion(image_pred, data_dict['image'])
                        batch_metrics['lpips'] = lpips_loss

                        loss += stage_config.loss_weight.lpips * lpips_loss

                    ## mse
                    if stage_config.loss_weight.mse:
                        mse_loss = self.mse_criterion(image_pred, data_dict['image'])
                        batch_metrics['mse'] = mse_loss

                        loss += stage_config.loss_weight.mse * mse_loss

                    ## encoder_latent_deviation
                    if stage_config.loss_weight.encoder_latent_deviation:
                        encoder_latent_deviation_loss = self.encoder_latent_deviation_criterion(self.inferer.latent, encoder_latent)
                        batch_metrics['encoder_latent_deviation'] = encoder_latent_deviation_loss

                        loss += stage_config.loss_weight.encoder_latent_deviation * encoder_latent_deviation_loss

                    ## generator_params_deviation
                    if stage_config.loss_weight.generator_params_deviation:
                        generator_params_deviation_loss = self.generator_params_deviation_criterion(
                            utils.common.flatten_parameters(self.inferer.g_ema.parameters()),
                            generator_params_inital
                        )
                        batch_metrics['generator_params_deviation'] = generator_params_deviation_loss

                        loss += stage_config.loss_weight.generator_params_deviation * generator_params_deviation_loss

                    ## ntexture_deviation
                    if stage_config.loss_weight.ntexture_deviation:
                        ntexture_deviation_loss = self.ntexture_deviation_criterion(
                            ntexture_pred,
                            initial_ntexture
                        )
                        batch_metrics['ntexture_deviation'] = ntexture_deviation_loss

                        loss += stage_config.loss_weight.ntexture_deviation * ntexture_deviation_loss


                    ## face_lpips
                    image_real_face = image_pred_face = None
                    if stage_config.loss_weight.face_lpips:
                        ### crop faces and get valik mask (some images don't contain faces)
                        image_real_face, valid_mask = self.crop_face_with_landmarks(
                            data_dict['image'], data_dict['landmarks'], face_size=(256, 256)
                        )

                        image_pred_face, _ = self.crop_face_with_landmarks(  # valid_mask is the same as for real
                            image_pred, data_dict['landmarks'], face_size=(256, 256)
                        )

                        face_lpips_loss = self.face_lpips_criterion(image_pred_face, image_real_face, valid_mask=valid_mask)

                        batch_metrics['face_lpips'] = face_lpips_loss

                        loss += stage_config.loss_weight.face_lpips * face_lpips_loss

                    ## unary_discriminator_feature_matching
                    if stage_config.loss_weight.unary_discriminator_feature_matching:
                        unary_discriminator_feature_matching_loss = self.unary_discriminator_feature_matching_criterion(
                            {'rgb': image_pred, 'segm': segm_pred, 'landmarks': data_dict['landmarks']},
                            {'rgb': data_dict['image'], 'segm': data_dict['segm'], 'landmarks': data_dict['landmarks']}
                        )['loss']
                        batch_metrics['unary_discriminator_feature_matching'] = unary_discriminator_feature_matching_loss

                        loss += stage_config.loss_weight.unary_discriminator_feature_matching * unary_discriminator_feature_matching_loss

                    ## unary_face_discriminator_feature_matching
                    if stage_config.loss_weight.unary_face_discriminator_feature_matching:
                        face_indices = np.arange(data_dict['image'].shape[0])

                        unary_face_discriminator_feature_matching_output_dict = self.unary_face_discriminator_feature_matching_criterion(
                            {'rgb': image_pred[face_indices], 'segm': segm_pred[face_indices], 'landmarks': data_dict['landmarks'][face_indices]},
                            {'rgb': data_dict['image'][face_indices], 'segm': data_dict['segm'][face_indices], 'landmarks': data_dict['landmarks'][face_indices]}
                        )

                        unary_face_discriminator_feature_matching_loss = unary_face_discriminator_feature_matching_output_dict['loss']
                        batch_metrics['unary_face_discriminator_feature_matching'] = unary_face_discriminator_feature_matching_loss

                        loss += stage_config.loss_weight.unary_face_discriminator_feature_matching * unary_face_discriminator_feature_matching_loss

                        face_image_pred = unary_face_discriminator_feature_matching_output_dict['pred_output_dict']['face']
                        face_image_target = unary_face_discriminator_feature_matching_output_dict['target_output_dict']['face']
                    else:
                        face_image_pred = None
                        face_image_target = None
                        
                    ## total loss
                    batch_metrics['total_loss'] = loss.item()

                    # calculate metrics
                    with torch.no_grad():
                        # mae
                        mae_metric = self.mae_criterion(image_pred, data_dict['image'])
                        batch_metrics['mae'] = mae_metric

                        # ssim
                        ssim_metric = self.ssim_criterion(image_pred, data_dict['image'])
                        batch_metrics['ssim'] = ssim_metric

                    # collect metrics
                    batch_metrics = utils.common.reduce_loss_dict(batch_metrics)
                    batch_metrics = utils.common.squeeze_metrics(batch_metrics)
                           
                    # optimization step
                    if mode == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    # update state
                    if mode == 'train':
                        self.state['train']['n_batches_passed'] += 1
                        self.state['train']['n_samples_passed'] += len(train_dict['image'])
                    
                    # save sbs and rotations to save_dir
                    if i == stage_config.n_iters - 1:
                        canvases = vis_images_sbs(data_dict['image'], image_pred, n_samples=1)
                        canvas = np.concatenate(canvases, axis=0)
                        cv2.imwrite(os.path.join(save_dir, 'sbs', f'{mode}_{stage}.png'), cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))

                        canvases = vis_rotation(
                            self.inferer,
                            None if self.inferer.latent is None else torch.cat([self.inferer.latent] * len(data_dict['image']), dim=0),
                            self.inferer.noise,
                            data_dict['verts'],
                            None if self.inferer.ntexture is None else torch.cat([self.inferer.ntexture] * len(data_dict['image']), dim=0),
                            data_dict['calibration_matrix'],
                            white=self.config.runner.white,
                            n_rotation_samples=self.config.log.log_n_rotation_samples,
                            n_samples=1
                        )
                        canvases = np.array(list(map(lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2RGB), canvases)))

                        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
                        video=cv2.VideoWriter(os.path.join(save_dir, 'rotations', f'{mode}_{stage}.mp4'), fourcc, 10, canvases[0].shape[:2])

                        for img in canvases:
                            video.write(img)

                        cv2.destroyAllWindows()
                        video.release()

        # update actual ntexture
        if self.inferer.latent is not None and self.inferer.noise is not None:
            with torch.no_grad():
                infer_output_dict = self.inferer.infer_pass(
                    torch.cat([self.inferer.latent] * len(train_dict['image']), dim=0),
                    train_dict['verts'],
                    noises=self.inferer.noise,
                    ntexture=None,
                    uv=None,
                    return_latents=False,
                    input_is_latent=True,
                    infer_g_ema=True
                )

                self.inferer.ntexture = nn.Parameter(infer_output_dict['fake_ntexture'].detach().clone(), requires_grad=True)
                self.inferer.latent = None
                self.inferer.noise = None