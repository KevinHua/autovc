"""
Generate speaker embeddings and metadata for training
"""
import os
import pickle
from model_bl import D_VECTOR
from collections import OrderedDict
import numpy as np
import torch

C = D_VECTOR(dim_input=80, dim_cell=768, dim_emb=256).eval()
if torch.cuda.is_available():
    C = C.cuda()
    
c_checkpoint = torch.load('dvector-ckpt.tar',map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

'''new_state_dict = OrderedDict()
for key, val in c_checkpoint.items():
    new_key = key[7:]
    new_state_dict[new_key] = val
C.load_state_dict(new_state_dict)
'''
C.load_state_dict(c_checkpoint['state_dict'])

num_uttrs = 10
len_crop = 128

# Directory containing mel-spectrograms
rootDir = './spmel'
dirName, subdirList, _ = next(os.walk(rootDir))
print('Found directory: %s' % dirName)


speakers = []

for speaker in sorted(subdirList):
    print('Processing speaker: %s' % speaker)
    utterances = []
    test_org = None
    emb_org = None

    utterances.append(speaker)
    _, _, fileList = next(os.walk(os.path.join(dirName,speaker)))

    # make speaker embedding
    if len(fileList) < num_uttrs:
        print('{} files is less than required uttr number {}'.format(speaker, num_uttrs))
        exit(1)

    idx_uttrs = np.random.choice(len(fileList), size=num_uttrs, replace=False)
    embs = []

    for i in range(num_uttrs):
        tmp = np.load(os.path.join(dirName, speaker, fileList[idx_uttrs[i]]))
        if i == 0:
            test_org = tmp
            print(test_org.shape)
        # only first utterance
        #else:
        #    test_org = np.append(test_org, tmp, axis=0)
        #    print(test_org.shape)

        candidates = np.delete(np.arange(len(fileList)), idx_uttrs)
        # choose another utterance if the current one is too short
        while tmp.shape[0] < len_crop:
            idx_alt = np.random.choice(candidates)
            tmp = np.load(os.path.join(dirName, speaker, fileList[idx_alt]))
            candidates = np.delete(candidates, np.argwhere(candidates==idx_alt))
        left = np.random.randint(0, tmp.shape[0]-len_crop)
        melsp = torch.from_numpy(tmp[np.newaxis, left:left+len_crop, :]).cuda()
        emb = C(melsp)
        embs.append(emb.detach().squeeze().cpu().numpy())
        
        if i == 0:
            emb_org = emb.detach().squeeze().cpu().numpy()

    # voice embedding and wave melsp
    utterances.append(np.mean(embs, axis=0))
    #utterances.append(emb_org)
    utterances.append(test_org)
    
    # create file list
    for fileName in sorted(fileList):
        utterances.append(os.path.join(speaker,fileName))
    speakers.append(utterances)
    
with open(os.path.join(rootDir, 'train.pkl'), 'wb') as handle:
    pickle.dump(speakers, handle)

