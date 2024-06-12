import pyiqa
import torch.nn
import kornia

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
    elif loss_name == 'Sobel':
        return Sobel()
    elif loss_name == 'Sobel_SSIM_structure':
        return Sobel_SSIM_structure(**kwargs)
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

class Sobel:
    def __init__(self):
        self.l = torch.nn.MSELoss()
    
    def __call__(self, source, target):
        source_edge = kornia.filters.sobel((source + 1.) / 2.)
        print(f'source_max: {source.max()}, source_edge max : {source_edge.max()}')
        target_edge = kornia.filters.sobel((target + 1.) / 2.)
        loss = self.l(source_edge, target_edge)
        print(f'Sobel loss : {loss}')
        return loss
    
class Sobel_SSIM_structure:
    def __init__(self, **kwargs):
        if 'get_ssim_map' in kwargs:
            self.get_ssim_map = kwargs['get_ssim_map']
        self.l = pyiqa.create_metric('ssim_structure', device='cuda', as_loss=True, **kwargs)
    
    def __call__(self, source, target):
        source_edge = kornia.filters.sobel((source + 1.) / 2.)
        target_edge = kornia.filters.sobel((target + 1.) / 2.)
        if hasattr(self, 'get_ssim_map') and self.get_ssim_map is True:
            ssim_val, ssim_map = self.l(source_edge, target_edge)
            self.ssim_map = ssim_map
            return 1. - ssim_val
        return 1. - self.l(source_edge, target_edge)