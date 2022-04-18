from pydub.silence import detect_nonsilent
from moviepy.editor import *
from pydub import AudioSegment
from pydub.silence import detect_nonsilent, split_on_silence
from scipy.spatial.distance import pdist
from math import *
from vp_track.infer_contrast import test,cos_sim
import numpy as np
import array
import librosa
import os

tiger=[]
# filePath:想被识别人的声纹库
filePath = '/Users/lingoace/Desktop/老虎/'
wav_list=os.listdir(filePath)
for i in range(len(wav_list)):
    wav_list[i] = "/Users/lingoace/Desktop/老虎/" + str(wav_list[i])
    if wav_list[i]=="/Users/lingoace/Desktop/老虎/.DS_Store":
        continue
    else:
        data=test(wav_list[i],read="file")
        # 计算声纹特征 并存储
        tiger.append(data)

# 存放视频中的声纹特征
mp4_vp=[]
# 存放时间
mp4_time=[]

# 视频分段并且提取声纹特征
mp4_file = "/Users/lingoace/Desktop/211350_7570520138866298_highlight.mp4"
audio = AudioSegment.from_file(mp4_file, "mp4")
audio = audio.set_frame_rate(16000)
audio = audio.set_channels(1)
audio = audio.set_sample_width(2)
silence_thresh=-45
chunks = detect_nonsilent(audio, min_silence_len=300, silence_thresh=silence_thresh, seek_step=10)  # 结果数组起止点单位为ms
min_silence_len=1000

# 提取视频中的声纹特征
for i, item in enumerate(chunks):
    waveform = audio[item[0]-10:item[1]+10]  # ms

    waveform_1 = np.array(array.array(waveform.array_type, waveform._data)).astype("float")
    extended_wav = np.append(waveform_1, waveform_1)
    if np.random.random() < 0.3:
        extended_wav = extended_wav[::-1]
    if extended_wav.size==0:
        continue
    linear = librosa.stft(extended_wav, n_fft=512, win_length=400, hop_length=160)
    mag, _ = librosa.magphase(linear)
    freq, freq_time = mag.shape
    if freq_time < 257:
        continue
    else:
        data = test(waveform_1)
        mp4_time.append((str(item[0]-10)+"-"+str(item[1]+10)))
        mp4_vp.append(data)

for i in range(len(tiger)):
    for j in range(len(mp4_time)):
        distance=cos_sim(tiger[i],mp4_vp[j])

        if distance>=0.5:
            print("tiger：{}  |  start_time:{}  |  end_time:{}  |  distance:{}".format(i,mp4_time[j].split("-")[0],mp4_time[j].split("-")[1],distance))


