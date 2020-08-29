import sys
import string
import warnings
import math
import os
import random
from collections import Counter

from multiprocessing import Pool

import torch
import torchtext

import numpy as np
import pandas as pd

import av

import lxml, lxml.html
import html

import spacy
import youtube
import pytube

from tqdm import tqdm


class YouTubeDataset(torch.utils.data.IterableDataset):
    """
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
    If ``key`` is ``VIDEO``, then the duration of ``AUDIO`` and ``TEXT`` fields are 
    determined by the ``video_cliplen`` which is the number of frames to present. 
    
    If using ``CC_TEXT`` as the key, you will need to get the max text duration from the ``max_text_dur`` property,
    and pad the data to the max duration in the video_transform and audio_transform.


    Parameters:
        root_dir (string): Root directory of the YouTube Dataset.
        channel (string):  Name to identify dataset locally
        split (string): The split to present - splits are determined by ``splits`` dictionary
        fields (list, optional): A list of ``fieldnames`` representing the data fields presented by the iterator
        key (string, optional): The key field's ``fieldname``
        video_format (string, optional): The video pixel format one of: 'yuv420p','rgb24', 'rgba'
        video_framerate (int, optional): if specified, it will resample the video
            so that it has `frame_rate`, and then the video_cliplen will be defined on the resampled video
        video_cliplen (int, optional): number of frames in a clip
        video_stridelen (int, tuple, optional): number of frames between each clip, if tuple a range for the stride (1,199) random between 1 and 199
        video_transform (callable, optional): A function/transform that  takes in a TxHxWxC video
            and returns a transformed version.
        image_transform (callable, optional): A function/transform that  takes in a CxHxW image
            and returns a transformed version.
        audio_format (string): The audio sample format, one of: 'u8', 's16', 's32', 'flt','dbl'
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
        dataset_fraction (float): fractuion of dataset to use (0. - 1.)
    Returns:
        tuple of data fields from dataset filtered by ``fields`` and ``key``. 
        ``fields`` list items will return data in the following formats:  
            ``VIDEOID`` (string): The YouTube video ID
            ``TITLE`` (string): The title of the video 
            ``DESCRIPTION`` (string): The description of the video 
            ``TIME`` (float): The presentation time in seconds of the returned data ``key`` 
            ``DURATION`` (float): The duration of the returned data ``key`` in seconds
            ``CC_TEXT`` (string): Close Caption Text
            ``IF_DATA`` (Tensor[C, H, W]): Image Tensor 'H' - Height, 'W' - Width, 'C' Channels
            ``VF_DATA`` (Tensor[T, H, W, C]): 'T' Video Frames or single frame
            ``AF_DATA`` (Tensor[K, L]): Audio Frames where `K` is the number of audio channels, and 'L' is samples
    """
        
    F_VIDEOID = 'VIDEOID'
    F_TITLE = 'TITLE'
    F_DESCRIPTION = 'DESCRIPTION'
    F_TIME = 'TIME'
    F_DURATION = 'DURATION'
    F_VF_DATA = 'VIDEO'
    F_IF_DATA = 'IMAGE'
    F_AF_DATA = 'AUDIO'
    F_CC_TEXT = 'TEXT'

    
    def __init__(self, root_dir, channel, split='train', fields=['IMAGE','TIME'], key='IMAGE',
                 video_format='rgb24', video_framerate=None, video_cliplen=None, video_stridelen=None, image_transform=None, video_transform=None,
                 audio_format='flt', audio_layout='mono', audio_sr=16000,  audio_cliplen=None, audio_transform=None,
                 text_lang='en', text_transform=None, dataset_fraction=1.0,
                 download=False, api_key=None, channel_id=None, user_name=None, splits={'train':0.80, 'test':0.20}):

        
        assert (key in fields), "YouTubeDataset: fields must contain key"     
        assert not download or (download and api_key != None), "YouTubeDataset: download requires api_key from Google API Console with Youtube access" 
        assert not download or (download and (channel_id != None or user_name != None)), "YouTubeDataset: download requires one of channel_id or user_name"
        assert key in fields, "YouTubeDataset: key must be in fields"
        assert (key == YouTubeDataset.F_VF_DATA and video_cliplen > 1) or True, "YouTubeDataset: video_cliplen must be 1 or greater if key is 'VIDEO'"
        assert (key == YouTubeDataset.F_AF_DATA and audio_cliplen > 1) or True, "YouTubeDataset: audio_cliplen must be 1 or greater if key is 'AUDIO'"

        if download:
            dl = YouTubeDatasetDownloader(root_dir, channel, splits, api_key, caption_lang=text_lang, channel_id=channel_id, user_name=user_name)
            dl.download_sync()
        
        
        csv_file = os.path.join(root_dir, channel, split + '.csv')

        assert os.path.isfile(csv_file), "YouTubeDataset: File Not Found " + csv_file
        
        self.dataframe = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.split = split
        self.fields = fields
        self.key = key
        self.channel = channel
        self.image_transform = image_transform
        self.video_transform = video_transform
        self.audio_transform = audio_transform
        self.text_transform = text_transform
        self.text_lang = text_lang
        
        if dataset_fraction > 0.0 and dataset_fraction < 1.0:
            self.dataframe=self.dataframe.sample(frac=dataset_fraction)

        self.start = 0
        self.size = self.dataframe.shape[0]
        self.video_format = video_format
        self.video_framerate = video_framerate
        self.video_cliplen = video_cliplen
        self.video_stridelen=video_stridelen
        self.audio_format = audio_format
        self.audio_sr = audio_sr
        self.audio_layout = audio_layout
        self.audio_cliplen = audio_cliplen
        
        self.raw_text_cache = None
        
        self._len = None
        
        

    def __iter__(self):
        return YTDSIterator(self)

    @property    
    def length(self):
        if self._len == None:
            if self.key == 'TEXT':
                self._len = len(self.raw_text)
            elif self.key == 'VIDEO' or self.key == 'IMAGE':
                if self.video_stridelen is None:
                    stride = self.video_cliplen
                elif isinstance(self.video_stridelen, tuple):
                    stride = random.randint(*self.video_stridelen)
                else:
                    stride = self.video_stridelen
                   
                self._len = int((sum([dur for pts, dur, text in self.raw_text]) * (self.video_framerate if self.video_framerate != None else 24)) // stride)
            elif self.key == 'AUDIO':
                self._len = int((sum([dur for pts, dur, text in self.raw_text]) * (self.video_framerate if self.video_framerate != None else 24)) //
                                (self.audio_cliplen if self.audio_cliplen != None else 512))
        return self._len
    
                    
    @property
    def fieldnames(self):
        """Get the list of valid ``fieldnames``.
        Returns:
            list: list of valid fieldnames
        """
        return [YouTubeDataset.F_VIDEOID, YouTubeDataset.F_TITLE, YouTubeDataset.F_DESCRIPTION, YouTubeDataset.F_TIME, 
                YouTubeDataset.F_DURATION, YouTubeDataset.F_VF_DATA, YouTubeDataset.F_IF_DATA, YouTubeDataset.F_AF_DATA, 
                YouTubeDataset.F_CC_TEXT]
    

    @property
    def raw_text(self):
        """Returns the raw TEXT of the dataset as a list
        Returns:
            list: [(TIME, DURATION, TEXT) ...] the Caption portion of the dataset
        """
        if self.raw_text_cache == None:
            self.raw_text_cache = [ (pts, dur, text) for pts, dur, text in YouTubeDataset(self.root_dir, self.channel, self.split, 
                                                             fields=[YouTubeDataset.F_TIME,YouTubeDataset.F_DURATION,YouTubeDataset.F_CC_TEXT],
                                                             key=YouTubeDataset.F_CC_TEXT) ]
        return self.raw_text_cache
    
    @property
    def max_text_dur(self):
        """Returns the max duration of the TEXT fields.
        Returns:
            float: duration in seconds
        """
        
        return max([dur for pts, dur, text in self.raw_text])


    @property
    def max_text_len(self):
        """Returns the max length of the TEXT fields in chars.
        Returns:
            int: length in characters
        """
        return max([len(text) for pts, dur, text in self.raw_text])
    
    def max_text_tokens(self,  tokenizer, vocab=None):
        """Returns the max number of tokens in the TEXT fields, using the specified totchtext tokenizer
        Returns:
            int: length in tokens
        """
        if vocab is None:
            return max([len(tokenizer(text)) for pts, dur, text in self.raw_text])
        else:
            return max([len([t for t in tokenizer(text) if str(t) in vocab.stoi]) for pts, dur, text in self.raw_text])
    
    @staticmethod    
    def build_vocab(root_dir, channel, vocab='vocab.txt', language="en", pretrained="fasttext.en.300d", pos=None, lemmatize=False):
        """Returns a new torchtext.experimental.vectors.Vectors object for the dataset, either from the vectors file or
           building a new one from the dataset using the tokenizer and the pretrained vectors
        Params:
            vocab: file name in the dataset directory of the cached vocab 
            tokenizer: tokenizer function
            pretrained: one of available pretrained vectors: charngram.100d fasttext.en.300d fasttext.simple.300d glove.42B.300d
                        glove.840B.300d glove.twitter.27B.25d glove.twitter.27B.50d glove.twitter.27B.100d glove.twitter.27B.200d g
                        love.6B.50d glove.6B.100d glove.6B.200d glove.6B.300d
            pos: parts of speech to include in vocab, None for all words or, list of Universal POS tags: 
                    'ADJ','ADP','PUNCT','ADV','AUX','SYM','INTJ','CONJ','X','NOUN','DET','PROPN','NUM','VERB','PART','PRON','SCONJ'
        Returns: 
            torchtext.experimental.vectors.Vectors
        """
        cache_path = os.path.join(root_dir, channel, vocab)
        
        if not os.path.exists(cache_path):
            assert pretrained in torchtext.vocab.pretrained_aliases, "build_vector: allowed pretrained vectors values are " + str(torchtext.vocab.pretrained_aliases.keys())
            counter = Counter()
            pt = torchtext.vocab.pretrained_aliases[pretrained]()

            nlp = spacy.load(language)
            
            if pos is None and not lemmatize:
                tokenizer = lambda text: [t.text for t in nlp.tokenizer(text) if t.text in pt.stoi]
            else:
                tokenizer = lambda text: [t.text if not lemmatize else t.lemma_ for t in nlp(text) if t.text in pt.stoi and t.pos_ in pos]
            
            
            for split in ['train', 'test', 'validate']:
                for [text] in tqdm(YouTubeDataset(root_dir, channel, split, fields=[YouTubeDataset.F_CC_TEXT], key=YouTubeDataset.F_CC_TEXT), desc=split):
                    counter.update(tokenizer(text))

            toks = {tok: ix for ix, tok in enumerate(counter)}
            vecs = [pt[tok] for tok in counter]

            with open(cache_path, 'w') as fp:
                fp.writelines([",".join([t, str(counter[t]), " ".join([*[str(float(d)) for d in vecs[toks[t]]], "\n"])]) for t in toks])
                
        # Open the Vector   
        with open(cache_path, 'r') as fp:
            toks={}
            vecs=[]
            counter=Counter()
        
            for ix, line in enumerate(fp.readlines()):
                key,count,vec = line.split(',')
                toks[key] = ix
                counter[key] = int(count)
                vecs.append(torch.tensor([float(v) for v in vec.split(' ') if v != "\n"]))
            vocab = torchtext.vocab.Vocab(counter)
            vocab.set_vectors(toks, vecs, 300)
            return vocab
    
            
    @staticmethod
    def worker_init(worker_id):
        """Worker Init Function for DataLoader. See torch.utils.data.DataLoader
        Parameters:
            worker_id (int): The ID of the worker
        """
        worker_info = torch.utils.data.get_worker_info()
        dataset = worker_info.dataset  # the dataset copy in this worker process
        overall_start = dataset.start
        overall_size = dataset.size
        worker_id = worker_info.id

        if overall_size < worker_info.num_workers:
            raise ValueError('Number of workers must be less than number of videos in split ('+str(overall_size)+')')

        # configure the dataset to only process the split workload
        per_worker = overall_size // worker_info.num_workers
        dataset.start = overall_start + worker_id * per_worker
        dataset.size = min(per_worker, overall_size - dataset.start) 


class CCFrame():
    def __init__(self, time, duration, text):
        self.__time = time
        self.__duration = duration
        self.__text = text

    @property
    def time(self):
        return self.__time

    @property
    def duration(self):
        return self.__duration

    @property
    def text(self):
        return self.__text

    def __repr__(self):
        return '<%s pts=%d, duration=%d, text=%s at 0x%x>' % (
            self.__class__.__name__,
            self.time,
            self.duration,
            self.text,
            id(self)
        )  

    @staticmethod
    def ParseCC(caption_path):
        with open(caption_path, 'r') as myfile:
            data = myfile.read()

        forest = lxml.html.fragments_fromstring(data, no_leading_text=True)

        transcript = []

        for e in forest:
            if e.tag == 'transcript':
                for text in e:
                    if text.tag == 'text':
                        try:
                            transcript.append(CCFrame(float(text.attrib['start']), float(text.attrib['dur']), html.unescape(text.text)))
                        except:
                            pass

        return transcript  

# This dataset returns the following tuple for each iteration
# pts as float
# video_frame as numpy array - the video frame at this pts
# audio_frame as numpy array - the entire audio frame for this pts
# caption_frame as string    - the caption text for this pts

class YTDSIterator():
    def __init__(self, dataset):
        self.dataset = dataset

        self.active_ix = -1
        self.active = None

        self.next_active()

    def next_active(self):
        try:
            # Get The Active Record from the dataset
            if self.active_ix == -1:
                self.active_ix = self.dataset.start
            else:
                self.active_ix = self.active_ix + 1

            if self.active_ix >= (self.dataset.start + self.dataset.size):
                raise StopIteration()
            
            self.active = self.dataset.dataframe.iloc[self.active_ix]

            self.video_id = self.active['video_id']
            self.title = self.active['title']  
            self.description = self.active['description']     

            video_path = os.path.join(self.dataset.root_dir,self.dataset.channel, 'video', self.video_id + '.mp4')
            caption_path = os.path.join(self.dataset.root_dir,self.dataset.channel, 'caption', self.video_id + '.xml')

            # Initialize Video 
            if (YouTubeDataset.F_VF_DATA in self.dataset.fields or 
                YouTubeDataset.F_IF_DATA in self.dataset.fields):                        
                self.vcontainer = av.open(video_path)
                self.video = self.vcontainer.decode(video=0)
                self.video_rate = self.vcontainer.streams.video[0].average_rate
            else:
                self.vcontainer = None
                self.video = None
                self.video_rate = 0

            # Initialize Audio 
            if (YouTubeDataset.F_AF_DATA in self.dataset.fields):                        
                self.acontainer = av.open(video_path)
                self.audio = self.acontainer.decode(audio=0)
                if (self.acontainer.streams.audio[0].sample_rate != self.dataset.audio_sr or 
                    self.acontainer.streams.audio[0].layout != self.dataset.audio_layout or
                    self.acontainer.streams.audio[0].format != self.dataset.audio_format):
                    self.audio_resampler = av.audio.resampler.AudioResampler(self.dataset.audio_format, 
                                                                             self.dataset.audio_layout, self.dataset.audio_sr)
                else:
                    self.audio_resampler = None
            else:
                self.acontainer = None
                self.audio = None
                self.audio_resampler = None

            # init captions
            if YouTubeDataset.F_CC_TEXT in self.dataset.fields:
                self.caption = iter(CCFrame.ParseCC(caption_path))
            else:
                self.caption = None
                
            # initialize VideoFrame start
            self.vclip = []
            self.vclip_time = 0.0

            # Initialize Audioframe Start
            self.aclip_time = 0.0
            self.aclip = []

            # Initialize Caption Start
            self.cclip_time = 0.0
            self.cclip_duration = 0.0
            self.cclip = []

            # set up the result
            self.result = {
                YouTubeDataset.F_VIDEOID: self.video_id,
                YouTubeDataset.F_TITLE: self.title,
                YouTubeDataset.F_DESCRIPTION: self.description
            }
        except StopIteration as si:
            raise si
            
        except Exception as e:
            print('Exception', type(e), e.args, e)
            raise StopIteration()
        
    def __iter__(self):
        return self


    def __next__(self):
        try:
            result = self.result
            
            
            if self.dataset.key == YouTubeDataset.F_CC_TEXT:
                clip_duration = self.cclip_duration
                clip_time = self.cclip_time
                
            if self.dataset.key == YouTubeDataset.F_VF_DATA:
                if self.dataset.video_framerate == None:
                    clip_duration = self.dataset.video_cliplen / self.video_rate
                else:
                    clip_duration = self.dataset.video_cliplen / self.dataset.video_framerate
                clip_time = self.vclip_time
            
            if self.dataset.key == YouTubeDataset.F_IF_DATA:
                clip_duration = 1. / self.video_rate
                clip_time = self.vclip_time

            if self.dataset.key == YouTubeDataset.F_AF_DATA:
                clip_duration = self.dataset.audio_cliplen / self.dataset.audio_sr
                clip_time = self.aclip_time
                

            if YouTubeDataset.F_CC_TEXT in self.dataset.fields:                        
                
                if self.dataset.key == YouTubeDataset.F_CC_TEXT:
                    # Get Next Caption
                    cc = next(self.caption)
                    self.cclip = [cc]
                    self.cclip_time = clip_time = cc.time
                    self.cclip_duration = clip_duration = cc.duration
                    cc_text = cc.text
                    
                    result[YouTubeDataset.F_TIME] = self.cclip_time
                    result[YouTubeDataset.F_DURATION] = self.cclip_duration

                
                else:    
                    # remove old text clips
                    for c in self.cclip:
                        if (c.time + c.duration) < clip_time:
                            self.cclip.remove(c)
                            if len(self.cclip) > 0:
                                self.cclip_time = self.cclip[0].time
                            else:
                                self.cclip_time = 0
                            self.cclip_duration = sum([c.duration for c in self.cclip])

                    # add new text clips
                    while self.cclip_duration < clip_duration:
                        cc = next(self.caption)
                        self.cclip.append(cc)
                        self.cclip_time = self.cclip[0].time
                        self.cclip_duration = sum([c.duration for c in self.cclip])
                    
                    cc_text = " ".join([c.text for c in self.cclip])
                    
                if self.dataset.text_transform:
                    cc_text = self.dataset.text_transform(cc_text)
                    
                result[YouTubeDataset.F_CC_TEXT] = cc_text


            if YouTubeDataset.F_VF_DATA in self.dataset.fields:                        
                # get the video segment
                if self.dataset.video_cliplen is None:
                    if self.dataset.video_framerate == None:
                        video_cliplen = int(clip_duration * self.video_rate)
                    else:
                        video_cliplen = int(clip_duration * self.dataset.video_framerate)
                else:
                    video_cliplen = self.dataset.video_cliplen

                if self.dataset.video_stridelen is None:
                    stride = video_cliplen
                elif isinstance(self.dataset.video_stridelen, tuple):
                    stride = random.randint(*self.dataset.video_stridelen)
                else:
                    stride = self.dataset.video_stridelen

                try:
                    if self.dataset.video_framerate == None:
                        video_chunklen = max(stride,video_cliplen) + 1
                    else:
                        video_chunklen = max(stride,video_cliplen) * self.video_rate // self.dataset.video_framerate + 1

                    while len(self.vclip) < video_chunklen:
                        vf = next(self.video)
                        #vf_data = torch.from_numpy(np.transpose(vf.to_ndarray(format=self.dataset.video_format), (2, 0, 1))) # (C, H, W) Image Tensor
                        vf_data = torch.from_numpy(vf.to_ndarray(format=self.dataset.video_format)).permute(2, 0, 1) # (C, H, W) Image Tensor
                        if self.dataset.image_transform:
                            vf_data = self.dataset.image_transform(vf_data)
                        self.vclip.append((vf_data, vf.time))

                except StopIteration:
#                    if len(self.vclip) == 0:
                    raise StopIteration()            

                if self.dataset.video_framerate != None and self.dataset.video_framerate != self.video_rate:
                    stride = stride * self.video_rate // self.dataset.video_framerate 
                    num_frames = len(self.vclip) * self.dataset.video_framerate // self.video_rate
                    idx = YTDSIterator._resample_video_idx(num_frames, self.video_rate, self.dataset.video_framerate)
                    if isinstance(idx, slice):
                        vclip = self.vclip[idx]
                    else:
                        vclip = [ self.vclip[i] for i in idx ]
                else:
                    vclip = self.vclip
                    
                #vf_data = torch.stack([ i[0] for i in vclip[:video_cliplen]]).permute(1, 0, 2, 3) # (C,T,H,W) - Video Tensor   
                vf_data = torch.stack([ i[0] for i in vclip[:video_cliplen]]).permute(0, 2, 3, 1) # (T,H,W,C) - Video Tensor   
                #vf_data = torch.stack([ i[0] for i in vclip[:video_cliplen]]) # (T,C,H,W) - Video Tensor   

                if self.dataset.video_transform:
                    vf_data = self.dataset.video_transform(vf_data)
                result[YouTubeDataset.F_VF_DATA] = vf_data    

                self.vclip_time = vclip[0][1]

                self.vclip = vclip[stride:]
                    
                if self.dataset.key == YouTubeDataset.F_VF_DATA:
                    result[YouTubeDataset.F_TIME] = clip_time = self.vclip_time
                    result[YouTubeDataset.F_DURATION] = clip_duration


            if YouTubeDataset.F_IF_DATA in self.dataset.fields:                        
                    
                if self.dataset.video_stridelen is None:
                    stride = 1
                elif isinstance(self.dataset.video_stridelen, tuple):
                    stride = random.randint(*self.dataset.video_stridelen)
                else:
                    stride = self.dataset.video_stridelen

                # get the video frame or segment
                for i in range(stride):
                    vf = next(self.video)
                vf_data = torch.from_numpy(np.transpose(vf.to_ndarray(format=self.dataset.video_format), (2, 0, 1))) # (C, H, W) Image Tensor
                if self.dataset.image_transform:
                    vf_data = self.dataset.image_transform(vf_data)
                result[YouTubeDataset.F_IF_DATA] = vf_data    
                self.vclip_time = vf.time
                    
                if self.dataset.key == YouTubeDataset.F_IF_DATA:
                    result[YouTubeDataset.F_TIME] = clip_time = self.vclip_time
                    result[YouTubeDataset.F_DURATION] = clip_duration
                    
                    
            if YouTubeDataset.F_AF_DATA in self.dataset.fields:
                
                if self.dataset.audio_cliplen is None:
                    audio_cliplen = int(clip_duration * self.dataset.audio_sr)
                else:
                    audio_cliplen = self.dataset.audio_cliplen

                # remove old audio clips
                for ix, (c, t)  in enumerate(self.aclip):
                    if (t + c.shape[-1] / self.dataset.audio_sr) < clip_time:
                        self.aclip.pop(ix)
                        if len(self.aclip) > 0:
                            self.aclip_time = self.aclip[0][1]
                        else:
                            self.aclip_time = 0
                
                # add new audio as needed
                while sum([c.shape[-1] for c, t in self.aclip]) < audio_cliplen:
                    af = next(self.audio)
                    
                    if self.audio_resampler:
                        af.pts = None
                        af = self.audio_resampler.resample(af)
                        
                    # get the audio frame as a numpy array and append to aclip
                    self.aclip.append((torch.from_numpy(af.to_ndarray(format=self.dataset.audio_format)), af.time))
                
                af_data = torch.cat([ c for c, t in self.aclip],dim=1) # (C,S) - Audio Tensor   
                
                self.aclip = [ (af_data[:,:audio_cliplen], self.aclip[0][1])]
                # reshape aclip to make first segment audio_cliplen in size
                if af_data.shape[-1] > audio_cliplen: 
                    self.aclip.append((af_data[:,audio_cliplen:], self.aclip_time + audio_cliplen / self.dataset.audio_sr))
        
                af_data = self.aclip[0][0]
                if self.dataset.audio_transform:
                    af_data = self.dataset.audio_transform(af_data)
                       
                result[YouTubeDataset.F_AF_DATA] =  af_data
                if self.dataset.key == YouTubeDataset.F_AF_DATA:
                    result[YouTubeDataset.F_TIME] = self.aclip[0][1] 
                    result[YouTubeDataset.F_DURATION] = clip_duration
                
            return [result[k] for k in self.dataset.fields]

        except StopIteration:
            self.next_active()
            return self.__next__()
        
    @staticmethod
    def _resample_video_idx(num_frames, original_fps, new_fps):
        step = float(original_fps) / new_fps
        if step.is_integer():
            # optimization: if step is integer, don't need to perform
            # advanced indexing
            step = int(step)
            return slice(None, None, step)
        idxs = torch.arange(num_frames, dtype=torch.float32) * step
        idxs = idxs.floor().to(torch.int64)
        return idxs

        
def YTDSVideoDownloadThunk(t):
    return t[0].download_video(t[1])
    

class YouTubeDatasetDownloader():
    def __init__(self, root_dir, channel, splits, api_key, 
                 channel_id=None, user_name=None, caption_lang="en", types=['VIDEO', 'CAPTION'], youtube_url = 'http://youtube.com/watch?v='):
    
        assert channel_id != None or user_name != None, "channel_id or user_name must be specified" 

        assert sum([splits[k] for k in splits]) == 1.0, "sum of splits must equal 1.0"
        
        
        self.root_dir = root_dir
        self.channel = channel
        self.ytapi = youtube.API(client_id=None, client_secret=None, api_key=api_key)
        self.channel_id = channel_id
        self.user_name = user_name
        self.splits = splits
        self.youtube_url = youtube_url
        self.caption_lang = caption_lang
        self.types=types
        
        self.channel_path = os.path.join(root_dir, channel)
        os.makedirs(self.channel_path, exist_ok=True)

        if 'VIDEO' in self.types:
            self.video_path = os.path.join(self.channel_path, 'video')
            os.makedirs(self.video_path, exist_ok=True)

        if 'CAPTION' in self.types:    
            self.caption_path = os.path.join(self.channel_path, 'caption')
            os.makedirs(self.caption_path, exist_ok=True)



    def channel_ids(self):
        if self.channel_id != None:
            channels = self.ytapi.get('channels', part='snippet,contentDetails', id=self.channel_id) 
        if self.user_name != None:
            channels = self.ytapi.get('channels', part='snippet,contentDetails', forUsername=self.user_name)
        return [ item['id'] for item in channels['items'] ]

 
    @staticmethod
    def playlist_sort_key(a):
        return a[1]

    def playlist_ids(self, channel_ids):
        playlist_ids = []
        for channel_id in channel_ids:

            nextPageToken=''

            while True:
                pl = self.ytapi.get('playlists', part='contentDetails', channelId=channel_id, pageToken=nextPageToken) 

                for it in pl['items']:
                    if it['kind'] == 'youtube#playlist':
                        playlist_ids.append((it['id'], int(it['contentDetails']['itemCount'])))

                if 'nextPageToken' not in pl.keys():
                    break
                else:
                    nextPageToken=pl['nextPageToken']

        playlist_ids.sort(key=self.playlist_sort_key, reverse=True)

        return playlist_ids


    def videos(self, playlist_ids):
        videos = []

        for playlist_id, playlist_size in tqdm(playlist_ids):
            nextPageToken=''
            while True:
                pli = self.ytapi.get('playlistItems', part='snippet,contentDetails', playlistId=playlist_id, pageToken=nextPageToken) 

                for it in pli['items']:
                    if it['kind'] == 'youtube#playlistItem':
                        snippet = it['snippet']
                        if snippet['resourceId']['kind'] == 'youtube#video':
                            videos.append([snippet['resourceId']['videoId'], snippet['title'], snippet['description']])

                if 'nextPageToken' not in pli.keys():
                    break
                else:
                    nextPageToken=pli['nextPageToken']

        return videos
            

    def make_dataframe(self, videos):
        self.df = pd.DataFrame(videos, columns =['video_id', 'title', 'description'], dtype = float)
        return self.df
    
    def parallel_download_videos(self, num_partitions=1, num_cores=1):
        df_split = np.array_split(self.df, num_partitions)
        selves = [self for _ in df_split] 
        
        pool = Pool(num_cores)
        newdf = pd.concat(pool.map(YTDSVideoDownloadThunk, zip(selves, df_split)))
        pool.close()
        pool.join()
        self.df = newdf


    def download_videos(self, subframe):
        drops = []
        for ix, video_id in enumerate(subframe['video_id']):

            url = self.youtube_url + video_id
            try:
                yt = pytube.YouTube(url)

                if yt.captions[self.caption_lang] != None:
                    #TODO: Download at a specific resolution - currently highest
                    if 'VIDEO' in self.types:
                        stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').asc().first()
                        video_filename = video_id #+ '.mp4' 
                        stream.download(output_path=self.video_path, filename=video_filename)
                    
                    if 'CAPTION' in self.types:
                        caption = yt.captions[self.caption_lang]
                        caption_filename = os.path.join(self.caption_path, video_id + ".xml")
                        with open(caption_filename, "w") as caption_file:
                            caption_file.write(caption.xml_captions)
                else:
                    drops.append(ix)
            
            except OSError as ex:
                raise ex
                
            except Exception as ex:
                print('pytube3 incompatibility, upgrade pytube3 to download: ' + url, 'Exception: ' + str(ex.__class__.__name__) + ', '+ str(ex))
                drops.append(ix)

            self.pbar.update()

        subframe = subframe.drop(drops)
        return subframe


    def make_splits(self):
        sz = self.df.shape[0]
        perm = np.random.permutation(range(sz))
        
        hwm = 0
        for k in self.splits:
            start = hwm
            end = min(hwm + int(self.splits[k] * sz), sz)
            hwm = end + 1
            
            split = self.df.iloc[perm[start:end]]
            split.to_csv(os.path.join(self.channel_path, k + '.csv'))
    

    def splits_exist(self):
        return all([ os.path.isfile(os.path.join(self.channel_path, k + '.csv')) for k in self.splits ])
        
    def download_sync(self):
        if not self.splits_exist():
            channel_ids = self.channel_ids()
            playlist_ids = self.playlist_ids(channel_ids)
            videos = self.videos(playlist_ids)
            self.make_dataframe(videos)

            self.pbar = tqdm(total=self.df.shape[0])
            #self.parallel_download_videos()
            self.df = self.download_videos(self.df)
            self.pbar.close()
            self.make_splits()
        else:
            print('Files already downloaded')