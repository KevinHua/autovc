import os
import pickle
import torch
import numpy as np
from math import ceil
from model_vc import Generator
import soundfile


def pad_seq(x, base=32):
    len_out = int(base * ceil(float(x.shape[0])/base))
    len_pad = len_out - x.shape[0]
    assert len_pad >= 0
    return np.pad(x, ((0,len_pad),(0,0)), 'constant'), len_pad

device = 'cuda' if torch.cuda.is_available() else 'cpu'
G = Generator(32,256,512,32).eval().to(device)
# original configure
# cause error: 
#G = Generator(16,256,512,16).eval().to(device)

g_checkpoint = torch.load('autovc-step1000000.ckpt', map_location=device)
G.load_state_dict(g_checkpoint['model'])

metadata = pickle.load(open('spmel/train.pkl', "rb"))

spect_vc = []

for sbmt_i in metadata:
    x_org = sbmt_i[2]
    x_org_data = np.reshape(x_org, x_org.size)
    
    x_org, len_pad = pad_seq(x_org)
    uttr_org = torch.from_numpy(x_org[np.newaxis, :, :]).to(device)
    emb_org = torch.from_numpy(sbmt_i[1][np.newaxis, :]).to(device)
    
    for sbmt_j in metadata:

        emb_trg = torch.from_numpy(sbmt_j[1][np.newaxis, :]).to(device)
        
        with torch.no_grad():
            _, x_identic_psnt, _ = G(uttr_org, emb_org, emb_trg)
            
        if len_pad == 0:
            uttr_trg = x_identic_psnt[0, 0, :, :].cpu().numpy()
        else:
            uttr_trg = x_identic_psnt[0, 0, :-len_pad, :].cpu().numpy()
        
        spect_vc.append( ('{}-{}'.format(sbmt_i[0], sbmt_j[0]), uttr_trg) )
        
        
#with open('results.pkl', 'wb') as handle:
#    pickle.dump(spect_vc, handle)

os.makedirs('melsp-conv', exist_ok=True)
trl = []
for sp in spect_vc:
    print(sp)

    wav_fn = os.path.join('melsp-conv', '{}-wave.npy'.format(sp[0]))
    wav_npy = sp[1]
    np.save(wav_fn, wav_npy)
