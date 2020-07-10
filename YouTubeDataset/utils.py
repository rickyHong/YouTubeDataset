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


def ipd_display_dataset(o, num_batches=1):
    """Displays a YouTubeDataset on all frontends.
    Parameters:
        o (YoutubeDataset or DataLoader): The YouTubeDataset or Dataloader(YouTubeDataset) to display
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
        
                if k == 'IMAGE':
                    if len(v.shape) == 4: # (B, C, H, W)
                        plt.figure(figsize=(32,32))
                        plt.axis("off")
                        plt.title(k)
                        plt.imshow(np.transpose(torchvision.utils.make_grid(v, padding=2).cpu(),(1,2,0)))  
                    elif len(v.shape) == 3: # (C,H,W)
                        plt.figure(figsize=(2,2))
                        plt.axis("off")
                        plt.title(k)
                        plt.imshow(np.transpose(v.cpu(),(1,2,0)))  
                                        
                if k == 'VIDEO':
                    if len(v.shape) == 5:   # (B, T, H, W, C)
                        v = v.view(v.shape[0]*v.shape[1], *v.shape[2:])
                    
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
                    if len(v.shape) == 3: # (B,K,L)
                        fig = plt.figure()
                        fig.set_figheight(v.shape[0])
                        fig.set_figwidth(10)
                        plt.title(k)
                        for i, a in enumerate(v):
                            if a.shape[0] == 1:
                                a = a.view(a.shape[1])

                            fig.add_subplot(v.shape[0]//2,2,i+1)
                            lrd.waveplot(a.numpy(), sr=ds.audio_sr)
                    elif len(v.shape) == 2: # (K,L)                        
                        if v.shape[0] == 1:
                            v = v.view(v.shape[1])
                    
                        plt.figure(figsize=(5, 2))
                        plt.title(k)
                        lrd.waveplot(v.numpy(), sr=ds.audio_sr)

            if (ix+1) == num_batches:
                return ipd.display(*ret)
