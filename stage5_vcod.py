import torch
import librosa
import pickle
from synthesis import build_model
from synthesis import wavegen
import soundfile

from torch.nn import functional as F

from hparams import hparams


spect_vc = pickle.load(open('results.pkl', 'rb'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = build_model().to(device)
checkpoint = torch.load("wavenet_checkpoint_372394.pth", map_location=device)
model.load_state_dict(checkpoint["state_dict"])

for spect in spect_vc:
    name = spect[0]
    c = torch.from_numpy(spect[1])
    print(name)

    print(c)

    if hparams['cin_pad'] > 0:
        c = F.pad(c, pad=(hparams['cin_pad'], hparams['cin_pad']), mode="replicate")

    waveform = wavegen(model, c=c)
    soundfile.write(name+'.wav', waveform, 16000)
