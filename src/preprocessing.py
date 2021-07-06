from pathlib import Path
import os
from scipy.fftpack import dct
import numpy as np
import librosa
import wave
import matplotlib.pyplot as plt
import math

dataset_path = Path('dataset/')

def endPointDetection(signal, sample_rate):
    
    
    frame_len = 256
    wave_energy = []
    top_k_frames = 10
    num_of_frame = int(math.ceil(len(signal) / frame_len))
    time_stamp = np.arange(0, num_of_frame) * (frame_len / sample_rate)
    biggest_energy = 0
    loop_continue_flag = True
    wave_point_idx = 0

    while loop_continue_flag:
        energy_of_a_frame = 0
        for point in range(int(wave_point_idx), int(wave_point_idx + frame_len)):
            if wave_point_idx + frame_len >= len(signal):
                loop_continue_flag = False
                energy_of_a_frame += 0
            else:
                energy_of_a_frame += signal[point] ** 2
        wave_energy.append(energy_of_a_frame)
        wave_point_idx += frame_len
    
    biggest_energy = max(wave_energy)
    threshold = 0.075 * biggest_energy + sum(wave_energy[0:top_k_frames:]) / top_k_frames

    epd_range = []
    epd_wave = []
    idx_of_energy = 0
    while idx_of_energy < len(wave_energy):
        detect_flag = False
        endure = True
        """ Find start point """
        while idx_of_energy < len(wave_energy):
            if wave_energy[idx_of_energy] > threshold:
                start_point = idx_of_energy * frame_len
                detect_flag = True
                break
            else:
                idx_of_energy += 1
        idx_of_energy += 1

        """ Find end point """
        while idx_of_energy < len(wave_energy):
            if wave_energy[idx_of_energy] < threshold:
                if endure:
                    endure = False
                    idx_of_energy += 1
                    continue
                else:
                    end_point = idx_of_energy * frame_len
                    idx_of_energy += 1
                    break            
            else:
                idx_of_energy += 1
        if detect_flag:
            epd_range.append([start_point, end_point])
            epd_wave.append(signal[start_point:end_point])
    return epd_range, epd_wave


    
    
    


def main():
    labels = os.listdir(dataset_path)
    for label in labels:
        signal, sample_rate = librosa.load(dataset_path / Path(label) / Path('%s_0_000.wav'%(label)), sr=None)
        epd_range, epd_wave = endPointDetection(signal, sample_rate)
        print(len(epd_range))



if __name__ == '__main__':
    main()