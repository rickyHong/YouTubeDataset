import torch

class Denormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def __call__(self, tensor):
        """
        Parameters:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


class Mosaicise(object):
    def __init__(self):
        pass
    
    def __call__(self, rgb):
        """
        Parameters:
            rgb (Tensor): Tensor image of size (C, H, W)
        Returns:
            Tensor:  Mosaicized tensor of size (1, H*scale, W*scale) bayer image.
        """
        _, w, h = rgb.shape
        # Create target array, twice the size of the original image
        raw = torch.zeros((1, w*2, h*2))
        # Map the RGB values in the original according to the BGGR pattern# 

        # R G R G 
        # G B G B 
        # R G R G 
        
        # Red
        raw[0, ::2, ::2] = rgb[0, :, :]

        # Green (top row of the Bayer matrix)
        raw[0, 1::2, ::2] = rgb[1, :, :]

        # Green (bottom row of the Bayer matrix)
        raw[0, ::2, 1::2] = rgb[1, :, :]

        # Blue
        raw[0, 1::2, 1::2] = rgb[2, :, :]

        return raw
    
class Demosaicise(object):
    def __init__(self):
        pass
    
    def __call__(self, raw):
        """
        Parameters:
            raw (Tensor): Tensor image of size (1, H, W) bayer image
        Returns:
            Tensor:  Mosaicized tensor of size (3, H*scale, W*scale) rgb image.
        """

        _, w, h = raw.shape

        rgb  = torch.zeros((3, w//2, h//2))
        
        #Red
        rgb[0:,:] = raw[0,::2,::2]
        
        #Green
        rgb[1:,:] = raw[0,1::2,::2]
        rgb[1:,:] = raw[0,::2,1::2]
        
        #Blue
        rgb[2:,:] = raw[0,1::2,1::2]
        
        return rgb
    
    
class RgbToYuv(object):
    def __init__(self):
        self.basis = torch.tensor([[ 0.29900, -0.16874,  0.50000],
                       [0.58700, -0.33126, -0.41869],
                       [ 0.11400, 0.50000, -0.08131]])
       
    
    def __call__(self, rgb):
        """
        Parameters:
            rgb (Tensor): Tensor image of size (3, H, W) rgb image
        Returns:
            Tensor:  Tensor of size (3, H, W) YUV image.
        """
        yuv = torch.dot(rgb,self.basis)
        
        if rgb.dtype == torch.unit8:
            yuv[:,:,1:]+= 128.0
            yuv = yuv.to(dtype=torch.uint8)
        else:
            yuv[:,:,1:]+= 0.5
            
        return yuv

class YuvToRgb(object):
    def __init__(self):
        self.basis = torch.tensor([[ 1.0, 1.0, 1.0],
                 [-0.000007154783816076815, -0.3441331386566162, 1.7720025777816772],
                 [ 1.4019975662231445, -0.7141380310058594 , 0.00001542569043522235]])
       
    
    def __call__(self, yuv):
        """
        Parameters:
            rgb (Tensor): Tensor image of size (3, H, W) rgb image
        Returns:
            Tensor:  Tensor of size (3, H, W) YUV image.
        """
        rgb = torch.dot(yuv,self.basis)
        
        if yuv.dtype == torch.unit8:
            rgb[:,:,0]-=179.45477266423404
            rgb[:,:,1]+=135.45870971679688
            rgb[:,:,2]-=226.8183044444304
            rgb = rgb.to(dtype=torch.uint8)

        else:
            rgb[:,:,0]-=179.45477266423404/255.0
            rgb[:,:,1]+=135.45870971679688/255.0
            rgb[:,:,2]-=226.8183044444304/255.0
            
        return rgb
    