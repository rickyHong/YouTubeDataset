import sys
import string
import warnings
import math
import os
import random
from multiprocessing import Pool

import torch
import numpy as np
import pandas as pd

import av

import lxml, lxml.html
import html

import youtube
import pytube

from tqdm import tqdm



class YouTubeDataset(torch.utils.data.IterableDataset):
    F_VIDEOID = 'VIDEOID'
    F_TITLE = 'TITLE'
    F_DESCRIPTION = 'DESCRIPTION'
    F_TIME = 'TIME'
    F_DURATION = 'DURATION'
    F_VF_DATA = 'VF_DATA'
    F_AF_DATA = 'AF_DATA'
    F_CC_TEXT = 'CC_TEXT'

    
    def __init__(self, root_dir, channel, split='train', fields=[F_VF_DATA,F_TIME], key=F_VF_DATA,
                 video_fmt='rgb24', video_cliplen=1, video_stridelen=-1, video_transform=None,
                 audio_fmt='s16', audio_layout='mono', audio_sr=44100,  audio_cliplen=-1, audio_transform=None,
                 text_lang='en', text_transform=None,
                 download=False, api_key=None, channel_id=None, user_name=None, splits={'train':0.90, 'test':0.1}):
        """
        Args:
            root_dir (string): Directory with all downloaded channels
            channel (string): Name of the Channel Dataset to load
            split (string, optional): The dataset split, supports train, test, or validate
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        assert YouTubeDataset.F_VF_DATA in fields or YouTubeDataset.F_TIME in fields, "YouTubeDataset: fields must contain F_VF_DATA and/or F_TIME"     
        assert not download or (download and api_key != None), "YouTubeDataset: download requires api_key from Google API Console with Youtube access" 
        assert not download or (download and (channel_id != None or user_name != None)), "YouTubeDataset: download requires one of channel_id or user_name"
        assert key in fields, "YouTubeDataset: key must be in fields"
        assert (key == YouTubeDataset.F_VF_DATA and video_cliplen != -1) or True, "YouTubeDataset: video_cliplen must be 1 or greater if key is 'VF_DATA'"
        assert (key == YouTubeDataset.F_AF_DATA and audio_cliplen != -1) or True, "YouTubeDataset: audio_cliplen must be 1 or greater if key is 'AF_DATA'"

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
        self.video_transform = video_transform
        self.audio_transform = audio_transform
        self.text_transform = text_transform
        self.text_lang = text_lang
        
        self.start = 0
        self.size = self.dataframe.shape[0]
        self.video_fmt = video_fmt
        self.video_cliplen = video_cliplen
        self.audio_fmt = audio_fmt
        self.audio_sr = audio_sr
        self.audio_layout = audio_layout
        self.audio_cliplen = audio_cliplen
        
        if video_stridelen == -1:
            self.video_stridelen=video_cliplen
        else:
            self.video_stridelen=video_stridelen

    def __iter__(self):
        return YTDSIterator(self)
            
    @staticmethod
    def worker_init(worker_id):
        worker_info = torch.utils.data.get_worker_info()
        dataset = worker_info.dataset  # the dataset copy in this worker process
        overall_start = dataset.start
        overall_size = dataset.size
        # configure the dataset to only process the split workload
        per_worker = int(math.ceil(overall_size / float(worker_info.num_workers)))
        worker_id = worker_info.id
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
        # Get The Active Record from the dataset
        if self.active_ix == -1:
            self.active_ix = self.dataset.start
        else:
            self.active_ix = self.active_ix + 1
        
        if self.active_ix >= self.dataset.start + self.dataset.size:
            raise StopIteration()
            
        self.active = self.dataset.dataframe.iloc[self.active_ix]

        self.video_id = self.active['video_id']
        self.title = self.active['title']  
        self.description = self.active['description']     

        video_path = os.path.join(self.dataset.root_dir,self.dataset.channel, 'video', self.video_id + '.mp4')
        caption_path = os.path.join(self.dataset.root_dir,self.dataset.channel, 'caption', self.video_id + '.xml')

        # Initialize Video and Audio
        container = av.open(video_path)
        self.video = container.decode(video=0)
        self.video_rate = container.streams.video[0].average_rate
        
        container = av.open(video_path)
        self.audio = container.decode(audio=0)

        if container.streams.audio[0].sample_rate != self.dataset.audio_sr:
            self.audio_resampler = av.audio.resampler.AudioResampler(self.dataset.audio_fmt, self.dataset.audio_layout, self.dataset.audio_sr)
        else:
            self.audio_resampler = None

        # init captions
        self.caption = iter(CCFrame.ParseCC(caption_path))

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

        
    def __iter__(self):
        return self


    def __next__(self):
        try:
            result = self.result
            
            
            if self.dataset.key == YouTubeDataset.F_CC_TEXT:
                clip_duration = self.cclip_duration
                clip_time = self.cclip_time
                
            if self.dataset.key == YouTubeDataset.F_VF_DATA:
                clip_duration = self.dataset.video_cliplen / self.video_rate
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


            if YouTubeDataset.F_VF_DATA in self.dataset.fields or YouTubeDataset.F_TIME in self.dataset.fields:                        
                        
                if isinstance(self.dataset.video_stridelen, tuple):
                    stride = random.randint(*self.dataset.video_stridelen)
                else:
                    stride = self.dataset.video_stridelen

                # get the video frame or segment
                if self.dataset.video_cliplen == 1:
                    for i in range(stride):
                        vf = next(self.video)
                    vf_data = np.transpose(vf.to_ndarray(format=self.dataset.video_fmt), (2, 0, 1)) # (C, H, W) Image Tensor
                    if self.dataset.video_transform:
                        vf_data = self.dataset.video_transform(vf_data)
                    result[YouTubeDataset.F_VF_DATA] = vf_data    
                    self.vclip_time = vf.time
                else:
                    if self.dataset.video_cliplen == -1:
                        video_cliplen = int(clip_duration * self.video_rate)
                    else:
                        video_cliplen = self.dataset.video_cliplen
                    
                    try:
                        while len(self.vclip) < video_cliplen:
                            vf = next(self.video)
                            vf_data = np.transpose(vf.to_ndarray(format=self.dataset.video_fmt), (2, 0, 1)) # (C, H, W) Image Tensor
                            if self.dataset.video_transform:
                                vf_data = self.dataset.video_transform(vf_data)
                            else:
                                vf_data = torch.from_numpy(vf_data)
                            self.vclip.append((vf_data, vf.time))

                    except StopIteration:
                        if len(self.vclip) == 0:
                            raise StopIteration()            

                    # vf_data = torch.stack([ i[0] for i in self.vclip]).permute(0, 2, 3, 1) # (T,H,W,C) - Video Tensor   
                    result[YouTubeDataset.F_VF_DATA] = torch.stack([ i[0] for i in self.vclip]) # (T,C,H,W) - Video Tensor   
                    self.vclip_time = self.vclip[0][1]

                    self.vclip = self.vclip[stride:]
                    
                if self.dataset.key == YouTubeDataset.F_VF_DATA:
                    result[YouTubeDataset.F_TIME] = clip_time = self.vclip_time
                    result[YouTubeDataset.F_DURATION] = clip_duration


            if YouTubeDataset.F_AF_DATA in self.dataset.fields:
                
                if self.dataset.audio_cliplen == -1:
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
                    self.aclip.append((torch.from_numpy(af.to_ndarray(format=self.dataset.audio_fmt)), af.time))
                
                af_data = torch.cat([ c for c, t in self.aclip],dim=1) # (C,S) - Audio Tensor   
                
                self.aclip = [ (af_data[:,:audio_cliplen], self.aclip[0][1])]
                self
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

        
def YTDSVideoDownloadThunk(t):
    return t[0].download_video(t[1])
    

class YouTubeDatasetDownloader():
    def __init__(self, root_dir, channel, splits, api_key, 
                 channel_id=None, user_name=None, caption_lang="en", youtube_url = 'http://youtube.com/watch?v='):
    
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
        
        
        self.channel_path = os.path.join(root_dir, channel)
        os.makedirs(self.channel_path, exist_ok=True)

        self.video_path = os.path.join(self.channel_path, 'video')
        self.caption_path = os.path.join(self.channel_path, 'caption')

        os.makedirs(self.video_path, exist_ok=True)
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

        for playlist_id, playlist_size in playlist_ids:
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
                    stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').asc().first()
                    video_filename = video_id #+ '.mp4' 
                    stream.download(output_path=self.video_path, filename=video_filename)
                    caption = yt.captions[self.caption_lang]
                    caption_filename = os.path.join(self.caption_path, video_id + ".xml")
                    with open(caption_filename, "w") as caption_file:
                        caption_file.write(caption.xml_captions)
                else:
                    drops.append(ix)

            except:
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