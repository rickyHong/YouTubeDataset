# YouTubeDataset

This project contains the YouTubeDataset for PyTorch. YouTubeDataset is a PyTorch IterableDataset, 
that can download a complete YouTube channel or users videos and cache them and present it as an IterableDataset.


## Installing YouTubeDataset

git clone https://github.com/secretlocation/YouTubeDataset.git
cd YouTubeDataset
python setup.py install


## Known Issues

YouTubeDataset uses pytube3 to download the YouTube videos. You may need to install a newer version either
from pypi or directly from github https://github.com/get-pytube/pytube3.git.


## Using YouTubeDataset

    from YouTubeDataset import YouTubeDataset

    # Download Secret Location YouTube channel and view it as the default, [ time, frame ]
    ds0 = YouTubeDataset('data', 'SecretLocation', 'train', 
                        download=True,
                        api_key=API_KEY,
                        user_name='thesecretlocation',
                        splits={'train':0.90, 'test':0.1})                  

    # print data sizes and value for first batch
    for batch in ds0:
        for k, v in zip(ds0.fields, batch):
            s = np.shape(v)
            if len(s) == 0:
                print(k,v)
            else:
                print(k,s)
        break


<div class="lm-Widget p-Widget lm-Panel p-Panel jp-Cell-outputWrapper"><div class="lm-Widget p-Widget jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser"><div class="jp-Collapser-child"></div></div><div class="lm-Widget p-Widget jp-OutputArea jp-Cell-outputArea" style=""><div class="lm-Widget p-Widget lm-Panel p-Panel jp-OutputArea-child"><div class="lm-Widget p-Widget jp-OutputPrompt jp-OutputArea-prompt"></div><div class="lm-Widget p-Widget jp-RenderedText jp-mod-trusted jp-OutputArea-output" data-mime-type="application/vnd.jupyter.stderr"><pre>  9%|▊         | 4/46 [00:15&lt;02:41,  3.85s/it]</pre></div></div><div class="lm-Widget p-Widget lm-Panel p-Panel jp-OutputArea-child"><div class="lm-Widget p-Widget jp-OutputPrompt jp-OutputArea-prompt"></div><div class="lm-Widget p-Widget jp-RenderedText jp-mod-trusted jp-OutputArea-output" data-mime-type="application/vnd.jupyter.stdout"><pre>pytube3 incompatibility, upgrade pytube3 to download: <a href="http://youtube.com/watch?v=E5E0Arr57YI" rel="noopener" target="_blank">http://youtube.com/watch?v=E5E0Arr57YI</a> Exception: KeyError, 'cipher'
</pre></div></div><div class="lm-Widget p-Widget lm-Panel p-Panel jp-OutputArea-child"><div class="lm-Widget p-Widget jp-OutputPrompt jp-OutputArea-prompt"></div><div class="lm-Widget p-Widget jp-RenderedText jp-mod-trusted jp-OutputArea-output" data-mime-type="application/vnd.jupyter.stderr"><pre> 11%|█         | 5/46 [00:16&lt;02:08,  3.14s/it]</pre></div></div><div class="lm-Widget p-Widget lm-Panel p-Panel jp-OutputArea-child"><div class="lm-Widget p-Widget jp-OutputPrompt jp-OutputArea-prompt"></div><div class="lm-Widget p-Widget jp-RenderedText jp-mod-trusted jp-OutputArea-output" data-mime-type="application/vnd.jupyter.stdout"><pre>pytube3 incompatibility, upgrade pytube3 to download: <a href="http://youtube.com/watch?v=wgl3nDpmXTQ" rel="noopener" target="_blank">http://youtube.com/watch?v=wgl3nDpmXTQ</a> Exception: KeyError, 'cipher'
</pre></div></div><div class="lm-Widget p-Widget lm-Panel p-Panel jp-OutputArea-child"><div class="lm-Widget p-Widget jp-OutputPrompt jp-OutputArea-prompt"></div><div class="lm-Widget p-Widget jp-RenderedText jp-mod-trusted jp-OutputArea-output" data-mime-type="application/vnd.jupyter.stderr"><pre> 52%|█████▏    | 24/46 [01:25&lt;01:24,  3.85s/it]</pre></div></div><div class="lm-Widget p-Widget lm-Panel p-Panel jp-OutputArea-child"><div class="lm-Widget p-Widget jp-OutputPrompt jp-OutputArea-prompt"></div><div class="lm-Widget p-Widget jp-RenderedText jp-mod-trusted jp-OutputArea-output" data-mime-type="application/vnd.jupyter.stdout"><pre>pytube3 incompatibility, upgrade pytube3 to download: <a href="http://youtube.com/watch?v=PAI8wBkxyw4" rel="noopener" target="_blank">http://youtube.com/watch?v=PAI8wBkxyw4</a> Exception: KeyError, 'cipher'
</pre></div></div><div class="lm-Widget p-Widget lm-Panel p-Panel jp-OutputArea-child"><div class="lm-Widget p-Widget jp-OutputPrompt jp-OutputArea-prompt"></div><div class="lm-Widget p-Widget jp-RenderedText jp-mod-trusted jp-OutputArea-output" data-mime-type="application/vnd.jupyter.stderr"><pre> 54%|█████▍    | 25/46 [01:27&lt;01:07,  3.23s/it]</pre></div></div><div class="lm-Widget p-Widget lm-Panel p-Panel jp-OutputArea-child"><div class="lm-Widget p-Widget jp-OutputPrompt jp-OutputArea-prompt"></div><div class="lm-Widget p-Widget jp-RenderedText jp-mod-trusted jp-OutputArea-output" data-mime-type="application/vnd.jupyter.stdout"><pre>pytube3 incompatibility, upgrade pytube3 to download: <a href="http://youtube.com/watch?v=G0FJHb6Cqn4" rel="noopener" target="_blank">http://youtube.com/watch?v=G0FJHb6Cqn4</a> Exception: KeyError, 'cipher'
</pre></div></div><div class="lm-Widget p-Widget lm-Panel p-Panel jp-OutputArea-child"><div class="lm-Widget p-Widget jp-OutputPrompt jp-OutputArea-prompt"></div><div class="lm-Widget p-Widget jp-RenderedText jp-mod-trusted jp-OutputArea-output" data-mime-type="application/vnd.jupyter.stderr"><pre> 65%|██████▌   | 30/46 [01:44&lt;00:53,  3.33s/it]</pre></div></div><div class="lm-Widget p-Widget lm-Panel p-Panel jp-OutputArea-child"><div class="lm-Widget p-Widget jp-OutputPrompt jp-OutputArea-prompt"></div><div class="lm-Widget p-Widget jp-RenderedText jp-mod-trusted jp-OutputArea-output" data-mime-type="application/vnd.jupyter.stdout"><pre>pytube3 incompatibility, upgrade pytube3 to download: <a href="http://youtube.com/watch?v=y50B7DRNm70" rel="noopener" target="_blank">http://youtube.com/watch?v=y50B7DRNm70</a> Exception: KeyError, 'en'
</pre></div></div><div class="lm-Widget p-Widget lm-Panel p-Panel jp-OutputArea-child"><div class="lm-Widget p-Widget jp-OutputPrompt jp-OutputArea-prompt"></div><div class="lm-Widget p-Widget jp-RenderedText jp-mod-trusted jp-OutputArea-output" data-mime-type="application/vnd.jupyter.stderr"><pre> 67%|██████▋   | 31/46 [01:46&lt;00:43,  2.87s/it]</pre></div></div><div class="lm-Widget p-Widget lm-Panel p-Panel jp-OutputArea-child"><div class="lm-Widget p-Widget jp-OutputPrompt jp-OutputArea-prompt"></div><div class="lm-Widget p-Widget jp-RenderedText jp-mod-trusted jp-OutputArea-output" data-mime-type="application/vnd.jupyter.stdout"><pre>pytube3 incompatibility, upgrade pytube3 to download: <a href="http://youtube.com/watch?v=reQPwK774O0" rel="noopener" target="_blank">http://youtube.com/watch?v=reQPwK774O0</a> Exception: KeyError, 'en'
</pre></div></div><div class="lm-Widget p-Widget lm-Panel p-Panel jp-OutputArea-child"><div class="lm-Widget p-Widget jp-OutputPrompt jp-OutputArea-prompt"></div><div class="lm-Widget p-Widget jp-RenderedText jp-mod-trusted jp-OutputArea-output" data-mime-type="application/vnd.jupyter.stderr"><pre> 76%|███████▌  | 35/46 [01:58&lt;00:31,  2.86s/it]</pre></div></div><div class="lm-Widget p-Widget lm-Panel p-Panel jp-OutputArea-child"><div class="lm-Widget p-Widget jp-OutputPrompt jp-OutputArea-prompt"></div><div class="lm-Widget p-Widget jp-RenderedText jp-mod-trusted jp-OutputArea-output" data-mime-type="application/vnd.jupyter.stdout"><pre>pytube3 incompatibility, upgrade pytube3 to download: <a href="http://youtube.com/watch?v=1BUJU3_ozaE" rel="noopener" target="_blank">http://youtube.com/watch?v=1BUJU3_ozaE</a> Exception: KeyError, 'en'
</pre></div></div><div class="lm-Widget p-Widget lm-Panel p-Panel jp-OutputArea-child"><div class="lm-Widget p-Widget jp-OutputPrompt jp-OutputArea-prompt"></div><div class="lm-Widget p-Widget jp-RenderedText jp-mod-trusted jp-OutputArea-output" data-mime-type="application/vnd.jupyter.stderr"><pre> 89%|████████▉ | 41/46 [02:18&lt;00:14,  2.86s/it]</pre></div></div><div class="lm-Widget p-Widget lm-Panel p-Panel jp-OutputArea-child"><div class="lm-Widget p-Widget jp-OutputPrompt jp-OutputArea-prompt"></div><div class="lm-Widget p-Widget jp-RenderedText jp-mod-trusted jp-OutputArea-output" data-mime-type="application/vnd.jupyter.stdout"><pre>pytube3 incompatibility, upgrade pytube3 to download: <a href="http://youtube.com/watch?v=y50B7DRNm70" rel="noopener" target="_blank">http://youtube.com/watch?v=y50B7DRNm70</a> Exception: KeyError, 'en'
</pre></div></div><div class="lm-Widget p-Widget lm-Panel p-Panel jp-OutputArea-child"><div class="lm-Widget p-Widget jp-OutputPrompt jp-OutputArea-prompt"></div><div class="lm-Widget p-Widget jp-RenderedText jp-mod-trusted jp-OutputArea-output" data-mime-type="application/vnd.jupyter.stderr"><pre> 91%|█████████▏| 42/46 [02:19&lt;00:08,  2.17s/it]</pre></div></div><div class="lm-Widget p-Widget lm-Panel p-Panel jp-OutputArea-child"><div class="lm-Widget p-Widget jp-OutputPrompt jp-OutputArea-prompt"></div><div class="lm-Widget p-Widget jp-RenderedText jp-mod-trusted jp-OutputArea-output" data-mime-type="application/vnd.jupyter.stdout"><pre>pytube3 incompatibility, upgrade pytube3 to download: <a href="http://youtube.com/watch?v=itIZUupuHJ4" rel="noopener" target="_blank">http://youtube.com/watch?v=itIZUupuHJ4</a> Exception: RegexMatchError, get_ytplayer_config: could not find match for config_patterns
</pre></div></div><div class="lm-Widget p-Widget lm-Panel p-Panel jp-OutputArea-child"><div class="lm-Widget p-Widget jp-OutputPrompt jp-OutputArea-prompt"></div><div class="lm-Widget p-Widget jp-RenderedText jp-mod-trusted jp-OutputArea-output" data-mime-type="application/vnd.jupyter.stderr"><pre>100%|██████████| 46/46 [02:31&lt;00:00,  3.23s/it]</pre></div></div><div class="lm-Widget p-Widget lm-Panel p-Panel jp-OutputArea-child"><div class="lm-Widget p-Widget jp-OutputPrompt jp-OutputArea-prompt"></div><div class="lm-Widget p-Widget jp-RenderedText jp-mod-trusted jp-OutputArea-output" data-mime-type="application/vnd.jupyter.stdout"><pre>VF_DATA (3, 360, 640)
TIME 0.0
</pre></div></div><div class="lm-Widget p-Widget lm-Panel p-Panel jp-OutputArea-child"><div class="lm-Widget p-Widget jp-OutputPrompt jp-OutputArea-prompt"></div><div class="lm-Widget p-Widget jp-RenderedText jp-mod-trusted jp-OutputArea-output" data-mime-type="application/vnd.jupyter.stderr"><pre>
</pre></div></div></div></div>


## class YouTubeDataset(torch.utils.data.dataset.IterableDataset)

YouTubeDataset is a flexible video dataset creation tool.
It allows the downloading of a complete YouTube channel as a dataset.
 
Downloading a channel dataset with YouTubeDataset is easy, set the ``download`` flag to True, 
and set ``api_key`` to a Google Cloud Platform API Key with rights for the YouTube Data API V3, 
and the YouTube ``channel_id``. The download will start automatically, and cache the dataset
locally in ``root_dir``/``channel``.
 
To use the dataset, set the `fields` parameter to a list of fields that will be extracted
from the dataset and presented by the iterator, and the ``key`` parameter to what key field is.
 
``fields`` is a list of one or more of the following ``fieldnames``:
 
 
``key`` refers to the dataset key field. The key field determines the length of all the
other dataset fields.
If ``key`` is ``VF_DATA``, then the duration of ``AF_AUDIO`` and ``CC_TEXT`` fields are 
determined by the ``video_cliplen`` which is the number of frames to present. 
If ``video_cliplen`` is 1 then ``VF_DATA`` frames are presented as TorchVision 
image tensors Tensor[H, W, C], otherwise they are presented as TorchVision 
video tensors Tensor[T, H, W, C].
 
If using ``CC_TEXT`` as the key, you will need to get the max text duration from the ``max_text_dur`` property,
and pad the data to the max duration in the video_transform and audio_transform.
 
 
### Args:
    root_dir (string): Root directory of the YouTube Dataset.
    channel (string):  Name to identify dataset locally
    split (string): The split to present - splits are determined by ``splits`` dictionary
    fields (list, optional): A list of ``fieldnames`` representing the data fields presented by the iterator
    key (string, optional): The key field's ``fieldname``
    video_format (string, optional): The video pixel format one of: 'yuv420p','rgb24', 'rgba'
    video_framerate (int, optional): if specified, it will resample the video
        so that it has `frame_rate`, and then the clips will be defined on the resampled video
    video_cliplen (int, optional): number of frames in a clip
    video_stridelen (int, optional): number of frames between each clip
    video_transform (callable, optional): A function/transform that  takes in a TxHxWxC video
        and returns a transformed version.
    image_transform (callable, optional): A function/transform that  takes in a CxHxW image
        and returns a transformed version.
    audio_format (string): The audio sample format, one of: 'u8', 's16', 's32', 'f32','f64'
    audio_layout (string, int, optional): The audio channel layout, either an integer number of channels, or
        audio_layout can be one or several of the following notations,
        separated by '+' or '|':
        - the name of an usual channel layout (mono, stereo, 4.0, quad, 5.0,
          5.0(side), 5.1, 5.1(side), 7.1, 7.1(wide), downmix);
        - the name of a single channel (FL, FR, FC, LFE, BL, BR, FLC, FRC, BC,
          SL, SR, TC, TFL, TFC, TFR, TBL, TBC, TBR, DL, DR);
        - a number of channels, in decimal, followed by 'c', yielding
          the default channel layout for that number of channels (@see
          av_get_default_channel_layout);
        - a channel layout mask, in hexadecimal starting with "0x" (see the
          AV_CH_* macros).
        Example: "stereo+FC" = "2c+FC" = "2c+1c" = "0x7"
    audio_sr (int, optional): The sample rate of audio clips in samples per second.
    audio_cliplen (int, optional): number of samples in a clip
    audio_transform (callable, optional): A function/transform that takes in a KxL audio tensor
        and returns a transformed version.
    text_lang (string, optional): the language of the captions using ISO 639-1. Default "en". 
    text_transform (callable, optional): A function/transform that takes in a string of text
        and returns the embedded text as a Tensor of any size.
 
### Returns:
    tuple of data fields from dataset filtered by ``fields`` and ``key``. 
    ``fields`` list items will return data in the following formats:  
        ``VIDEOID`` (string): The YouTube video ID
        ``TITLE`` (string): The title of the video 
        ``DESCRIPTION`` (string): The description of the video 
        ``TIME`` (float): The presentation time in seconds of the returned data ``key`` 
        ``DURATION`` (float): The duration of the returned data ``key`` in seconds
        ``CC_TEXT`` (string): Close Caption Text
        ``VF_DATA`` (Tensor[H, W, C],Tensor[T, H, W, C]): 'T' Video Frames or single frame
        ``AF_DATA`` (Tensor[K, L]): Audio Frames where `K` is the number of audio channels, and 'L' is samples
        
        
## License
YouTubeDataset is MIT-style licensed, as found in the LICENSE file.
