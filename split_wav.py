#!/usr/bin/env python

import os
import sys
import math
import argparse

import librosa
import soundfile
import numpy as np


def split_wav(audio_path, output_root, sampling_rate, segment_seconds, multi_speakers, speaker_id):
    if multi_speakers:
        speakers = sorted(os.listdir(audio_path))
        for spk in speakers:
            spk_path = os.path.join(audio_path, spk)
            if os.path.isdir(spk_path):
                split_wav_by_speaker(spk_path, output_root, sampling_rate, segment_seconds, None)
    else:
        split_wav_by_speaker(audio_path, output_root, sampling_rate, segment_seconds, speaker_id)
        
def split_wav_by_speaker(audio_path, output_root, sampling_rate, segment_seconds, speaker_id):
    if not speaker_id:
        speaker_id = os.path.basename(audio_path)
    
    output_dir = '{output_root}/p{audio_id}'.format(output_root=output_root, audio_id=speaker_id)
    os.makedirs(output_dir, exist_ok=True)
    
    fl = sorted(os.listdir(audio_path));
    seg_index = 1
    for ft in fl:
        fn = os.path.join(audio_path, ft)
        if os.path.isfile(fn):            
            y, sr = librosa.load(fn, sampling_rate)
            if len(y) == 0:
                print('{} load failed.'.format(fn))
                continue
            
            segment_len = math.ceil(segment_seconds * sampling_rate)
            pos = 0
            while pos < len(y):
                seg = y[pos:pos + segment_len]
                
                seg_file = '{output_dir}/p{speaker_id}_{seg_index:03d}.wav'.format(output_dir=output_dir, speaker_id=speaker_id, seg_index=seg_index)
                soundfile.write(seg_file, seg, sampling_rate)        
                
                pos += segment_len
                seg_index += 1
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_path", type=str, help="A dirctory that contains all voice audios of a speaker when multi_speakers is 0. Or a containing folder of multiple speaker subfolders")
    parser.add_argument("--output_root", type=str, help="path to save splitted result waves")
    parser.add_argument("--sampling_rate", type=int, default=16000, required=False, help="the output wave's sampling rate, 16000 by default")
    parser.add_argument("--segment_seconds", type=int, default=5, required=False, help="duration in seconds of splitted wave segment. 5 seconds by default")
    parser.add_argument("--multi_speakers", type=int, default=1, help="1 - multiple speakers mode or 0 - one speaker mode. multi-mode by default")
    parser.add_argument("--speaker_id",  type=int, default=None, required=False, help="speaker id to make a directory that contains all splitted waves. valid only when multi_speakers equals 0. None by default")
    config = parser.parse_args()
    print(config)
    return config

if __name__ == '__main__':
    split_wav(**vars(parse_args()))
    