import torch

class Denormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def __call__(self, tensor):
        """
        Args:
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
        Args:
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
        Args:
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
    
