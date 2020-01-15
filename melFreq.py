import librosa
import librosa.display
from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav


(rate,sig) = wav.read("output.wav")
mfcc_feat = mfcc(sig,rate)
fbank_feat = logfbank(sig,rate)

#print(fbank_feat[1:3,:])


y, sr = librosa.load('output.wav')
librosa.feature.mfcc(y=y, sr=sr)
librosa.feature.mfcc(y=y, sr=sr, hop_length=1024, htk=True)
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,fmax=8000)
librosa.feature.mfcc(S=librosa.power_to_db(S))
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfccs, x_axis='time')
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()
plt.show()