import librosa
import numpy as np
from scipy import signal
from scipy.signal import find_peaks
from scipy.io import wavfile

import matplotlib.pyplot as plt
import glob
from dtw import accelerated_dtw

file = "voice//numbers"
suffix1 = ".wav"  # Suffix of file note that this program only supports wav files

SAVE_FILE = 0  # Save filtered audio to f8ile
SAVE_SEGMENTS = 1


def preprocess_audio_file(FS, x):
    try:
        if x.shape[1] == 2:
            x = x.T
            x = librosa.core.to_mono(x.astype(float))   # convert signal to mono-signal
    except IndexError:
        pass
    if FS != 8000:
        x = librosa.resample(x.astype(float), FS, 8000)
        FS = 8000
    return FS, x


def filter_audio(filename, band_pass_filter):
    FS, x = wavfile.read(filename + suffix1)  # Load the audio
    FS, x = preprocess_audio_file(FS, x)     # Change the sample rate
    b = signal.firwin(101, band_pass_filter, fs=FS, pass_zero=False)  # Create a filter
    x = signal.lfilter(b, [1], x, axis=0)  # Apply the filter
    if SAVE_FILE:
        wavfile.write(filename + "-f" + suffix1, FS, x.astype(np.int16))
    return FS, x


def segment(filename):
    suffix2 = ".wav"
    if suffix2 in filename:
        filename, suffix2 = filename.split('.')
        suffix2 = '.' + suffix2
    FS, x = filter_audio(filename, [200, 3800])

    NS = 10  # Window size in 10ms
    MS = 10  # Window offset in ms
    L = int(NS * (FS/1000))
    K = int(MS * (FS/1000))

#  Short term energy function
    energy = []
    m = 0
    np.seterr(divide='ignore')  # in case of division with 0, disable the potential error
    while True:
        short_term_energy = x[m*K:m*K+L].astype(np.int64)
        energy.append(np.sum(np.square(short_term_energy)))
        m += 1
        if m*K > len(x) or m*K+L > len(x):
            energy = np.array(energy)
            energy = 10*np.log(energy)
            energy = np.subtract(energy, energy.max())
            break

    peaks, _ = find_peaks(energy, distance=150)
    plt.plot(energy)
    plt.plot(peaks, energy[peaks], "o")
    plt.plot(energy, "green")
    np.seterr(divide='warn')
    offset = 0  # Offset in case the first few values in E are -inf which causes problems
    for val in energy:
        if val == -np.inf:
            offset += 1
        else:
            break
    energy_average = np.mean(energy[offset:10 + offset])
    energy_signal = np.std(energy[offset:10 + offset])   # std = standard deviation

    Lev = max(energy_average + 3 * energy_signal, -60)  # low level/threshold of energy

    end_idx = 0
    voice_pos = []

    # Find the multi-frames where the logarithmic energy is above Lev and save them in a list
    for i, j in enumerate(energy):
        if i < end_idx:
            continue
        if j >= Lev:
            start_point = i
            for j, k in enumerate(energy[i:]):
                if k <= Lev:
                    end_point = j + start_point
                    voice_pos.append([start_point, end_point])
                    break
    # Look through voice_pos and merge elements which are no longer than 100 frames long
    nextFrame = True
    find_sound_pos = []
    for i, tup in enumerate(voice_pos):
        if nextFrame:
            start_idx = tup[0]
            nextFrame = False
        if tup[1] - start_idx < 30:
            continue
        else:
            try:
                if voice_pos[i+1][1] - start_idx > 100:
                    end_idx = voice_pos[i][1]
                    find_sound_pos.append([start_idx, end_idx])
                    nextFrame = True
                    continue
            except IndexError:
                end_idx = voice_pos[i][1]
                find_sound_pos.append([start_idx, end_idx])
                break

    FS, x = wavfile.read(filename + suffix2)
    prefix, filename = filename.split('//')
    prefix += '//segmented//'
    find_sound_pos = np.multiply(find_sound_pos, K)

    if SAVE_SEGMENTS:
        for i, pos in enumerate(find_sound_pos):
            wavfile.write(prefix + filename + "-{}".format(i) + suffix2, FS, x[pos[0]:pos[1]].astype(np.int16))
    return find_sound_pos


prefix = "voice//"
audioname = "numbers"
suffix3 = ".wav"

SAVE = 0

if __name__ == "__main__":
    FS, x = wavfile.read(prefix + audioname + suffix3)  # Load the audio clip with the original samples
    FS, x = preprocess_audio_file(FS, x)
    print('This is the frequency: ', FS, '\nand this is the signal\n', x)
    NS = 10  # Window size in ms
    MS = 10  # Window offset in ms
    L = int(NS * (FS / 1000))
    K = int(MS * (FS / 1000))
    x_segments = segment(prefix + audioname + suffix3)

    for j, seg in enumerate(x_segments):
        feat = []
        y = x[seg[0]:seg[1]]
        i = 0
        hamming_wind = np.hamming(L)  # Make a hamming window
        while True:
            frame = np.multiply(y[i*K:i*K+L], hamming_wind)
            feat.append(librosa.feature.mfcc(frame.astype(float), n_mfcc=18)[1:])
            i += 1
            if len(y) - (i*K+L) == 0:
                break
            elif len(y) - (i*K+L) < 0:
                frame = np.multiply(y[i*K:], np.hamming(len(y)-i*K))
                feat.append(librosa.feature.mfcc(y[i*K:].astype(float), n_mfcc=18)[1:])
                break
        feat = np.array(feat)

        feat = np.reshape(feat, (feat.shape[0], feat.shape[1]))
        if SAVE:
            np.save(prefix+"mfcc//"+audioname+"-{}".format(j), feat)
        else:
            match = []
            match_file = []
            for it in glob.glob(prefix + 'mfcc//*'):
                if audioname in it:
                    continue
                z = np.load(it)
                match.append(accelerated_dtw(feat, z, dist='euclidean')[0])
                match_file.append(it[12:-4])
            print(match_file[match.index(min(match))][3:])
    out = segment(file)
    data = np.divide(segment(file), 8000)
    print(data)
    print("Results are displayed in seconds")
    # time1, time2 = data.T
    # plt.scatter(time1, time2)
    plt.show()




