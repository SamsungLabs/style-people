import smplx
import os

def build_smplx_model_dict(smplx_model_dir, device):
    gender2filename = dict(neutral='SMPLX_NEUTRAL.pkl', male='SMPLX_MALE.pkl', female='SMPLX_FEMALE.pkl')
    gender2path = {k:os.path.join(smplx_model_dir, v) for (k, v) in gender2filename.items()}
    gender2model = {k:smplx.body_models.SMPLX(v).to(device) for (k, v) in gender2path.items()}

    return gender2model