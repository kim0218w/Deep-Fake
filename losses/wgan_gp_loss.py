# losses/wgan_gp_loss.py
import torch
import torch.nn.functional as F

def d_logistic_loss(real_pred, fake_pred):
    return F.softplus(fake_pred).mean() + F.softplus(-real_pred).mean()

def g_nonsaturating_loss(fake_pred):
    return F.softplus(-fake_pred).mean()

def r1_penalty(real_pred, real_img):
    grad_real = torch.autograd.grad(
        outputs=real_pred.sum(), inputs=real_img,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    return grad_real.pow(2).view(grad_real.size(0), -1).sum(1).mean()
