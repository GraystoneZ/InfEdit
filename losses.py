import pyiqa
import torch.nn

def init_structure_loss(loss_name : str, **kwargs):
    if loss_name == 'L1':
        return torch.nn.L1Loss()
    elif loss_name == 'MSE':
        return torch.nn.MSELoss()
    elif loss_name == 'SSIM':
        return SSIM()
    elif loss_name == 'MS_SSIM':
        return MS_SSIM()
    elif loss_name == 'LPIPS':
        return LPIPS()
    elif loss_name == 'SSIM_structure':
        return SSIM_structure(**kwargs)
    else:
        print('Undefined structure loss')
        return None

class LPIPS:
    def __init__(self):
        self.l = pyiqa.create_metric('lpips', device='cuda', as_loss=True)
    
    def __call__(self, source, target):
        return self.l((source + 1.) / 2., (target + 1.) / 2.)

class SSIM:
    def __init__(self, **kwargs):
        self.l = pyiqa.create_metric('ssimc', device='cuda', as_loss=True, **kwargs)
    
    def __call__(self, source, target):
        return 1. - self.l((source + 1.) / 2., (target + 1.) / 2.)

class MS_SSIM:
    def __init__(self):
        self.l = pyiqa.create_metric('ms_ssim', device='cuda', as_loss=True)
    
    def __call__(self, source, target):
        return 1. - self.l((source + 1.) / 2., (target + 1.) / 2.)

class SSIM_structure:
    def __init__(self, **kwargs):
        if 'get_ssim_map' in kwargs:
            self.get_ssim_map = kwargs['get_ssim_map']
        self.l = pyiqa.create_metric('ssim_structure', device='cuda', as_loss=True, **kwargs)
    
    def __call__(self, source, target):
        if hasattr(self, 'get_ssim_map') and self.get_ssim_map is True:
            ssim_val, ssim_map = self.l((source + 1.) / 2., (target + 1.) / 2.)
            self.ssim_map = ssim_map
            return 1. - ssim_val
        return 1. - self.l((source + 1.) / 2., (target + 1.) / 2.)
