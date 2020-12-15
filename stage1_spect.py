import os
import pickle
import numpy as np
import soundfile as sf
from scipy import signal
from scipy.signal import get_window
from librosa.filters import mel
from numpy.random import RandomState

import autovc_melsp


# audio file directory
rootDir = './wavs'
# spectrogram directory
targetDir = './spmel'


dirName, subdirList, _ = next(os.walk(rootDir))
print('Found directory: %s' % dirName)

def person_int_id(str_id):
    prod = 1
    for c in str_id.upper():
        if 'A' <= c <= 'Z':
            code = ord(c) - ord('A') + 10
            prod = prod * 37 + code;
        elif '0' <= c <= '9':
            prod = prod * 37 + int(c)
        else:
            prod = prod * 37 + 36

    return prod % (2^32)

for subdir in sorted(subdirList):
    print(subdir)
    if not os.path.exists(os.path.join(targetDir, subdir)):
        os.makedirs(os.path.join(targetDir, subdir))
    _,_, fileList = next(os.walk(os.path.join(dirName,subdir)))
    prng = RandomState(person_int_id(subdir[1:])) 
    for fileName in sorted(fileList):
        # Read audio file
        x, fs = sf.read(os.path.join(dirName,subdir,fileName))
        S = autovc_melsp.log_melsp_01(x)
        
        # save spect    
        np.save(os.path.join(targetDir, subdir, fileName[:-4]),
                S.astype(np.float32), allow_pickle=False)
        
