import onnxruntime
import numpy as np
import torch
import librosa
import os
import time


# 加载并预处理音频
def load_audio(audio,mode='train',read="audio",sr=16000,win_length=400,hop_length=160, n_fft=512, spec_len=257):
    # 读取音频数据
    if read=="audio":
        wav = audio
    else:
        wav, sr_ret = librosa.load(audio, sr=sr)
    # 数据拼接
    if mode == 'train':
        extended_wav = np.append(wav, wav)
        if np.random.random() < 0.3:
            extended_wav = extended_wav[::-1]
    else:
        extended_wav = np.append(wav, wav[::-1])
    # 计算短时傅里叶变换
    linear = librosa.stft(extended_wav, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
    # 计算复数图谱的幅度值和相位值。
    mag, _ = librosa.magphase(linear)
    # 幅度值和相位值
    freq, freq_time = mag.shape

    # 幅度值>=257
    assert  freq_time >= spec_len, "非静音部分长度不能低于1.3s"

    if mode == 'train':
        # 随机裁剪
        rand_time = np.random.randint(0, freq_time - spec_len)
        spec_mag = mag[:, rand_time:rand_time + spec_len]
    else:
        spec_mag = mag[:, :spec_len]
    mean = np.mean(spec_mag, 0, keepdims=True)
    std = np.std(spec_mag, 0, keepdims=True)
    spec_mag = (spec_mag - mean) / (std + 1e-5)
    spec_mag = spec_mag[np.newaxis, :]

    return spec_mag

def load_model(path):
    # sess = onnxruntime.InferenceSession(path,providers=["CUDAExecutionProvider"])  # use gpu
    sess = onnxruntime.InferenceSession(path)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    return sess,input_name,output_name


# 预处理音频
def infer(audio,read):
    input_shape = eval('(1, 257, 257)')
    data = load_audio(audio,read=read, mode='infer', spec_len=input_shape[2])
    data = data[np.newaxis, :]
    data = torch.tensor(data, dtype=torch.float32)
    return data

def cos_sim(feature1,feature2):
    # 对角余弦值
    feature1=feature1.reshape(512)
    feature2=feature2.reshape(512)

    dist = np.dot(feature1, feature2) / (np.linalg.norm(feature1) * np.linalg.norm(feature2))
    return dist


def test(audio,read="audio"):
    # path: onnxmodel path
    path=os.path.abspath('/Users/lingoace/PycharmProjects/Lingoace/LingoAce-AI/classmonitor/VoiceprintRecognition/speech_class.onnx')
    sess,input_name,output_name=load_model(path)
    audio_data=infer(audio,read=read)
    prediction = sess.run([output_name], {input_name: audio_data.numpy()})
    return prediction[0]


if __name__ == '__main__':
    pass
