import os
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
from os.path import isfile, join

#-----------------------------------------------------------------------------------

##https://github.com/laishawadhwa/Multi-modal-music-genre-classification/blob/main/XGBOOST/feature_extraction.py
def extracting_trackid(filename):
    # Extract the track ID from the filename by removing the extension
    track_ids = os.path.splitext(filename)[0]
    return track_ids

def feature_extraction(filepath, Savepath):
    id = 0  # Song ID

    track_id_vector = []  # List to store track IDs
    tempo = []
    chroma_STFT_mean = []  # Chroma short time fourier transform
    chroma_STFT_std = []

    chroma_CQT_mean = []
    chroma_CQT_std = []

    chroma_CENS_mean = []
    chroma_CENS_std = []

    rms_mean = []
    rms_std = []

    zcr_mean = []
    zcr_std = []

    rolloff_mean = []
    rolloff_std = []

    melspectrogram_mean = []
    melspectrogram_std = []

    spect_centroid_mean = []
    spect_centroid_std = []

    spect_flatness_mean = []
    spect_flatness_std = []

    spect_contrast_mean = []
    spect_contrast_std = []

    spect_bandwidth_mean = []
    spect_bandwidth_std = []

    spect_flux_mean = []
    spect_flux_std = []

    tonnetz_mean = []
    tonnetz_std = []

    harmonic_mean = []
    harmonic_std = []

    perc_mean = []
    perc_std = []

    rhythm_mean = []
    rhythm_std = []

    pitch_hist_mean = []
    pitch_hist_std = []

    mfcc_mean = []
    mfcc_std = []

    mfccs_mean = [[] for _ in range(13)]
    mfccs_std = [[] for _ in range(13)]
    chroma_mean = [[] for _ in range(12)]
    chroma_std = [[] for _ in range(12)]

    # Traversing over each file in path
    file_data = [file for file in os.listdir(filepath) if os.path.isfile(os.path.join(filepath, file))]
    for each_line in file_data:
        if each_line[-4:].lower() == '.mp3':
            # Reading Song
            songname = os.path.join(filepath, each_line)
            # Append the track ID to the list using the extract_track_id function
            track_id_vector.append(extracting_trackid(each_line))


            # #load function takes songname as input & reads audiofile from specified path and returns two value y & sr
            y, sr = librosa.load(songname)
            signal, _ = librosa.effects.trim(y)
            y1, sr1 = sf.read(songname, always_2d=True)
            y1 = y1.flatten('F')[:y1.shape[0]]

            tempo_val, _ = librosa.beat.beat_track(y=signal)
            tempo.append(tempo_val)

            chroma_STFT = librosa.feature.chroma_stft(y=signal, sr=sr, window='hann')
            chroma_CQT = librosa.feature.chroma_cqt(y=signal, sr=sr)
            chroma_cens = librosa.feature.chroma_cens(y=signal)
            root_mean_square = librosa.feature.rms(y=signal)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(signal)
            rolloff = librosa.feature.spectral_rolloff(y=signal)
            melspectrogram = librosa.feature.melspectrogram(y=signal, window='hann')
            spect_centroid = librosa.feature.spectral_centroid(y=signal)
            spect_flatness = librosa.feature.spectral_flatness(y=signal)
            spect_contrast = librosa.feature.spectral_contrast(y=signal, sr=sr)
            spect_bandwidth = librosa.feature.spectral_bandwidth(y=signal)
            spect_flux = librosa.onset.onset_strength(y=signal, sr=sr)
            tonnetz = librosa.feature.tonnetz(y=signal, sr=sr)
            harmonic = librosa.effects.harmonic(signal)
            percussive = librosa.effects.percussive(signal)
            rhythm = librosa.feature.tempogram(y=signal, sr=sr)
            pitch_histogram = librosa.feature.tonnetz(y=signal)
            mfcc = librosa.feature.mfcc(y=y1, sr=sr, n_mfcc=13)

            # Transforming Features
            chroma_STFT_mean.append(np.mean(chroma_STFT))
            chroma_STFT_std.append(np.std(chroma_STFT))

            chroma_CQT_mean.append(np.mean(chroma_CQT))
            chroma_CQT_std.append(np.std(chroma_CQT))

            chroma_CENS_mean.append(np.mean(chroma_cens))
            chroma_CENS_std.append(np.std(chroma_cens))
            rms_mean.append(np.mean(root_mean_square))
            rms_std.append(np.std(root_mean_square))
            zcr_mean.append(np.mean(zero_crossing_rate))
            zcr_std.append(np.std(zero_crossing_rate))
            rolloff_mean.append(np.mean(rolloff))
            rolloff_std.append(np.std(rolloff))
            melspectrogram_mean.append(np.mean(melspectrogram))
            melspectrogram_std.append(np.std(melspectrogram))
            spect_centroid_mean.append(np.mean(spect_centroid))
            spect_centroid_std.append(np.std(spect_centroid))
            spect_flatness_mean.append(np.mean(spect_flatness))
            spect_flatness_std.append(np.std(spect_flatness))
            spect_contrast_mean.append(np.mean(spect_contrast))
            spect_contrast_std.append(np.std(spect_contrast))
            spect_bandwidth_mean.append(np.mean(spect_bandwidth))
            spect_bandwidth_std.append(np.std(spect_bandwidth))
            spect_flux_mean.append(np.mean(spect_flux))
            spect_flux_std.append(np.std(spect_flux))
            tonnetz_mean.append(np.mean(tonnetz))
            tonnetz_std.append(np.std(tonnetz))
            harmonic_mean.append(np.mean(harmonic))
            harmonic_std.append(np.std(harmonic))
            perc_mean.append(np.mean(percussive))
            perc_std.append(np.std(percussive))
            rhythm_mean.append(np.mean(rhythm))
            rhythm_std.append(np.std(rhythm))
            pitch_hist_mean.append(np.mean(pitch_histogram))
            pitch_hist_std.append(np.std(pitch_histogram))
            mfcc_mean.append(np.mean(mfcc))
            mfcc_std.append(np.std(mfcc))

            mfccs_mean2 = np.mean(mfcc, axis=1)
            mfccs_std2 = np.std(mfcc, axis=1)
            chroma_mean2 = np.mean(chroma_cens, axis=1)
            chroma_std2 = np.std(chroma_cens, axis=1)

            for i in range(13):
                mfccs_mean[i].append(np.mean(mfcc[i]))
                mfccs_std[i].append(np.std(mfcc[i]))

            for i in range(12):
                chroma_mean[i].append(np.mean(chroma_cens[i]))
                chroma_std[i].append(np.std(chroma_cens[i]))

    # Concatenating Features into one csv and json format
    feature_set = pd.DataFrame({
        'track_id': track_id_vector,
        'tempo': tempo,
        'chroma_stft_mean': chroma_STFT_mean,
        'chroma_stft_std': chroma_STFT_std,
        'chroma_cqt_mean': chroma_CQT_mean,
        'chroma_cqt_std': chroma_CQT_std,
        'chroma_cens_mean': chroma_CENS_mean,
        'chroma_cens_std': chroma_CENS_std,
        'rms_mean': rms_mean,
        'rms_std': rms_std,
        'zcr_mean': zcr_mean,
        'zcr_std': zcr_std,
        'rolloff_mean': rolloff_mean,
        'rolloff_std': rolloff_std,
        'melspectrogram_mean': melspectrogram_mean,
        'melspectrogram_std': melspectrogram_std,
        'spect_centroid_mean': spect_centroid_mean,
        'spect_centroid_std': spect_centroid_std,
        'spect_flatness_mean': spect_flatness_mean,
        'spect_flatness_std': spect_flatness_std,
        'spect_contrast_mean': spect_contrast_mean,
        'spect_contrast_std': spect_contrast_std,
        'spect_bw_mean': spect_bandwidth_mean,
        'spect_bw_std': spect_bandwidth_std,
        'spect_flux_mean': spect_flux_mean,
        'spect_flux_std': spect_flux_std,
        'tonnetz_mean': tonnetz_mean,
        'tonnetz_std': tonnetz_std,
        'harmonic_mean': harmonic_mean,
        'harmonic_std': harmonic_std,
        'percussive_mean': perc_mean,
        'percussive_std': perc_std,
        'rhythm_mean': rhythm_mean,
        'rhythm_std': rhythm_std,
        'pitch_hist_mean': pitch_hist_mean,
        'pitch_hist_std': pitch_hist_std,
        'mfcc_mean': mfcc_mean,
        'mfcc_std': mfcc_std,
    })

    for i in range(13):
        feature_set[f'mfccs_mean_{i}'] = mfccs_mean[i]
        feature_set[f'mfccs_std_{i}'] = mfccs_std[i]

    for i in range(12):
        feature_set[f'chroma_mean_{i}'] = chroma_mean[i]
        feature_set[f'chroma_std_{i}'] = chroma_std[i]

    feature_set.to_csv(Savepath, index=False)

directory_path = r'C:\Dissertation\Music4all_dataset\Audio_data'
save_csvfile = r'C:\Dissertation\Music4all_dataset\Audio_features.csv'

# Call the function to extract features and save to CSV
feature_extraction(directory_path, save_csvfile)

#spectral envelope, rhythme based, pitch histogram, spectral kurtosis


#https://www.researchgate.net/publication/346359767_Audio_Features_for_Music_Emotion_Recognition_a_Survey ***





















