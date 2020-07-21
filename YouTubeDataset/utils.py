import tempfile
import base64
import os

import torch
import torchvision

import numpy as np

import matplotlib.pyplot as plt
import IPython.display as ipd
import librosa.display as lrd

from YouTubeDataset import YouTubeDataset


def ipd_display_dataset(o, decoders=None, num_batches=1):
    """Displays a YouTubeDataset on all frontends.
    Parameters:
        o (YoutubeDataset or DataLoader): The YouTubeDataset or Dataloader(YouTubeDataset) to display
        decoders {'FORMAT': (Callable), ...}: a Dictionary of decoders, one per dataset field type 
        num_batches (int): the number of batches to display, default = 1
    Returns:
        handle (IPython.display.DisplayHandle): display handle
    """
    
    ds = dl = None
    if isinstance(o, torch.utils.data.IterableDataset):
        ds = dl = o

    if isinstance(o, torch.utils.data.DataLoader):
        dl = o
        ds = dl.dataset
    
    assert isinstance(ds, YouTubeDataset), "ipd_display_dataset only supports instances of YouTubeDataset"
    
    if ds.video_framerate != None:
        fps = ds.video_framerate
    else:
        fps = iter(ds).video_rate
    
    if ds != None and dl != None:
        print(ds.fields)
        
        if (decoders != None) and ('VIDEO' in ds.fields) and ('IMAGE' in decoders) and not ('VIDEO' in decoders):
            decoders['VIDEO'] = lambda v: torch.stack([decoders['IMAGE'](f) for f in v])
        
        ret = []
        
        for ix, batch in enumerate(dl):
            for k, v in zip(ds.fields, batch):
                
                if isinstance(v, torch.Tensor):
                    s = v.shape
                    t = v.dtype
                else:
                    t = type(v)
                    s = []
                
                if len(s) == 0:
                    print(k,t,v)
                else:
                    print(k,t,s)
        
                if decoders != None and k in decoders:
                    if isinstance(dl, torch.utils.data.DataLoader): 
                        v = [decoders[k](f) for f in v]
                        if len(v) > 0 and isinstance(v[0], torch.Tensor):
                            v = torch.stack(v)
                    else:
                        v = decoders[k](v)

                if k == 'IMAGE':
                    if isinstance(dl, torch.utils.data.DataLoader): 
                        plt.figure(figsize=(32,32))
                        plt.axis("off")
                        plt.title(k)
                        plt.imshow(np.transpose(torchvision.utils.make_grid(v, padding=2).cpu(),(1,2,0)))  
                    else:                                            # (C,H,W)
                        plt.figure(figsize=(2,2))
                        plt.axis("off")
                        plt.title(k)
                        plt.imshow(np.transpose(v.cpu(),(1,2,0)))  
                                        
                if k == 'VIDEO':
                    if isinstance(dl, torch.utils.data.DataLoader):    # (B, T, H, W, C) => (T, H, W, C)
                        v = v.reshape(v.shape[0]*v.shape[1], *v.shape[2:])
                    
                    if v.dtype != torch.uint8:
                        v = (v * 255).to(dtype=torch.uint8)
                    
                    try:
                        file = tempfile.mktemp() + '.mp4'   
                        torchvision.io.write_video(file, v, fps)
                        
                        with open(file,'rb') as fp:
                            data = str(base64.b64encode(fp.read()),'utf-8')
                        ret.append(ipd.Video(data=data, embed=True, mimetype='video/mp4'))

                    finally:
                        os.remove(file)
                        
                if k == 'AUDIO':
                    if isinstance(dl, torch.utils.data.DataLoader): 
                        # (B, K, L) => (K, L)
                        # (B, K, F, T) ==> (K, F, T)
                        v = v.permute(*range(1,len(v.shape)),0)
                        v = v.reshape(*v.shape[:-2], np.prod(v.shape[-2:]))
                    
                    if len(v.shape) == 2: # (K,L)  - Audio Tensor                      
                        if v.shape[0] == 1:
                            v = v.reshape(v.shape[0]*v.shape[1])
                        
                        n = np.asfortranarray(v.numpy())
                        plt.figure(figsize=(5, 2))
                        plt.title(k)
                        lrd.waveplot(n, sr=ds.audio_sr)
                        ret.append(ipd.Audio(n, rate=ds.audio_sr))

                    elif len(v.shape) == 3: # (K, F, T) - Mel Spectrogram Tensor
                        plt.figure(figsize=(10, 4))
                        plt.title(k)
                        lrd.specshow(v[0].numpy(), y_axis='linear', hop_length=200, x_axis='time', sr=ds.audio_sr)
                        
                if k == 'TEXT':
                    if isinstance(v, str):
                        print('TEXT', v)
                    elif ((isinstance(v, list) or isinstance(v, tuple)) and isinstance(v[0], str)):
                        print('TEXT', v)

            if (ix+1) == num_batches:
                return ipd.display(*ret)
