import os
from tqdm import tqdm
import numpy as np
import pickle
import cv2

import torch
from torch import nn

import lpips
import kornia

import utils
from utils.bbox import compute_bboxes_from_keypoints, crop_resize_image

from inference_module.criterion import LPIPSCriterion, MSECriterion, MAECriterion, SSIMCriterion, \
    UnaryDiscriminatorNonsaturatingCriterion, UnaryDiscriminatorFeatureMatchingCriterion
    

class Runner:
    def __init__(self, config, inferer, smplx_model, image_size, device='cuda:0'):
        self.config = config
        self.inferer = inferer
        self.smplx_model = smplx_model

        self.image_size = image_size
        self.input_size = image_size // 2


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

        ## face_lpips
        self.face_lpips_criterion = LPIPSCriterion(net='vgg').to(device)


    def _get_latent_optimization_group(self, lr):
        optimization_group = {
            'params': [self.inferer.latent],
            'lr': lr
        }
        return optimization_group

    def _get_generator_optimization_group(self, lr):
        generator_params = []
        
        # convs
        generator_params.extend(self.inferer.generator.conv1.parameters())
        for l in self.inferer.generator.convs:
            generator_params.extend(l.parameters())
        
        # to_rgbs
        generator_params.extend(self.inferer.generator.to_rgb1.parameters())
        for l in self.inferer.generator.to_rgbs:
            generator_params.extend(l.parameters())

        optimization_group = {
            'params': generator_params,
            'lr': lr
        }

        return optimization_group

    def _get_noise_optimization_group(self, lr):
        optimization_group = {
            'params': self.inferer.noises,
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
        else:
            raise NotImplementedError(f"Unknown name {name}")

        return optimization_group


    def crop_face_with_landmarks(self, image, landmarks, face_size=256):
        face_kp = landmarks[:, 25:93].clone()

        valid_mask = (face_kp[..., 0] < 0).sum(dim=1) <= 0
        invalid_mask = torch.logical_not(valid_mask)
        
        # add dummy valid landmarks for invalid images
        face_kp[invalid_mask] = torch.ones(
            torch.sum(invalid_mask), face_kp.shape[1], face_kp.shape[2], device=face_kp.device
        )
        
        bboxes_estimate = compute_bboxes_from_keypoints(face_kp)
        face = crop_resize_image(image, bboxes_estimate, face_size)
        
        return face, valid_mask

    def make_smplx(self, train_dict):
        smplx_output = self.smplx_model(
                    global_orient=train_dict['global_orient'],
                    transl=train_dict['transl'],
                    betas=self.inferer.betas,
                    expression=train_dict['expressions'],
                    body_pose=train_dict['body_pose'],
                    left_hand_pose=train_dict['left_hand_pose'][:, :6],
                    right_hand_pose=train_dict['right_hand_pose'][:, :6],
                    jaw_pose=train_dict['jaw_pose'],
                )

        return smplx_output

    def run_epoch(self, train_dict):
        # self.inferer.eval()

        # get smplx betas, hand and body pose
        self.inferer.betas = train_dict['betas']
        self.inferer.betas.requires_grad_(True)


        v_inds = np.load(self.config.runner.v_inds_path)

        # save initial generator params
        generator_params_inital = utils.common.flatten_parameters(
            self.inferer.generator.parameters()
        ).detach().clone()
        
        # init with encoder
        encoder_latent = self.inferer.encoder.forward(kornia.resize(train_dict['real_rgb'], (self.config.encoder.image_size , self.config.encoder.image_size )))
        encoder_latent = encoder_latent.mean(0, keepdim=True)  # TODO: more smart averaging of latent vectors
        self.inferer.latent.data = encoder_latent.detach().clone()

        # save initial ntexture

        with torch.no_grad():
            infer_output_dict = self.inferer.infer_pass(
                torch.cat([self.inferer.latent] * len(train_dict['real_rgb']), dim=0),
                train_dict['verts'],
                ntexture=None,
                uv=None
            )

        initial_ntexture = infer_output_dict['fake_ntexture'][:1].detach().clone()

        stages = sorted(self.config.stages.keys())
        for stage in stages:
            stage_config = self.config.stages[stage]

            # maybe switch input source
            if stage_config.input_source == 'ntexture':
                with torch.no_grad():
                    infer_output_dict = self.inferer.infer_pass(
                        torch.cat([self.inferer.latent] * len(train_dict['real_rgb']), dim=0),
                        train_dict['verts'],
                        ntexture=None,
                        uv=None
                    )

                    self.inferer.ntexture = nn.Parameter(infer_output_dict['fake_ntexture'][:1].detach().clone(), requires_grad=True)

                    self.inferer.latent = None
                    self.inferer.noises = None

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
                torch.autograd.set_grad_enabled(True)

                smplx_output = self.make_smplx(train_dict)

                verts_3d = smplx_output.vertices
                verts_3d = verts_3d[:, v_inds]

                verts_ndc, matrix_ndc = self.inferer.diff_uv_renderer.convert_to_ndc(
                    verts_3d,
                    train_dict['K'],
                    self.input_size, self.input_size,
                    near=0.01, far=20.0,
                    invert_verts=False
                )

                uv, uv_da, mask, rast = self.inferer.diff_uv_renderer.render(verts_ndc, matrix_ndc, render_h=self.input_size, render_w=self.input_size)



                infer_output_dict = self.inferer.infer_pass(
                    None if self.inferer.latent is None else torch.cat([self.inferer.latent] * len(train_dict['real_rgb']), dim=0),
                    None,
                    ntexture=None if self.inferer.ntexture is None else torch.cat([self.inferer.ntexture] * len(train_dict['real_rgb']), dim=0),
                    uv=uv
                )

                image_pred = infer_output_dict['fake_img']
                segm_pred = infer_output_dict['fake_segm']
                ntexture_pred = infer_output_dict['fake_ntexture']

                # calculate losses
                loss = 0.0

                ## lpips
                if stage_config.loss_weight.lpips:
                    lpips_loss = self.lpips_criterion(image_pred, train_dict['real_rgb'])
                    loss += stage_config.loss_weight.lpips * lpips_loss

                ## mse
                if stage_config.loss_weight.mse:
                    mse_loss = self.mse_criterion(image_pred, train_dict['real_rgb'])
                    loss += stage_config.loss_weight.mse * mse_loss

                ## encoder_latent_deviation
                if stage_config.loss_weight.encoder_latent_deviation:
                    encoder_latent_deviation_loss = self.encoder_latent_deviation_criterion(self.inferer.latent, encoder_latent)
                    loss += stage_config.loss_weight.encoder_latent_deviation * encoder_latent_deviation_loss

                ## generator_params_deviation
                if stage_config.loss_weight.generator_params_deviation:
                    generator_params_deviation_loss = self.generator_params_deviation_criterion(
                        utils.common.flatten_parameters(self.inferer.generator.parameters()),
                        generator_params_inital
                    )
                    loss += stage_config.loss_weight.generator_params_deviation * generator_params_deviation_loss

                ## ntexture_deviation
                if stage_config.loss_weight.ntexture_deviation:
                    ntexture_deviation_loss = self.ntexture_deviation_criterion(
                        ntexture_pred,
                        initial_ntexture
                    )
                    loss += stage_config.loss_weight.ntexture_deviation * ntexture_deviation_loss


                ## face_lpips
                image_real_face = image_pred_face = None
                if stage_config.loss_weight.face_lpips:
                    ### crop faces and get valik mask (some images don't contain faces)
                    image_real_face, valid_mask = self.crop_face_with_landmarks(
                        train_dict['real_rgb'], train_dict['landmarks'], face_size=256
                    )

                    image_pred_face, _ = self.crop_face_with_landmarks(  # valid_mask is the same as for real
                        image_pred, train_dict['landmarks'], face_size=256
                    )

                    face_lpips_loss = self.face_lpips_criterion(image_pred_face, image_real_face, valid_mask=valid_mask)
                    loss += stage_config.loss_weight.face_lpips * face_lpips_loss
                else:
                    face_image_pred = None
                    face_image_target = None
                    

                       
                print('loss', loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return self.inferer.ntexture