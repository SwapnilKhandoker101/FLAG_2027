
import argparse
import numpy as np
import torch
import torch.utils.data
from torch.autograd import Variable
import pandas as pd
from retrieval_model import FOP

# changes to make it here:
# ver = 'v1'                                                                   # OLD
ver = 'v4'                                                                     # CHANGED (v4)
heard_lang = 'Bangla'                                                         # 'English' or 'Bangla'
track = 'gender'                                                            # CHANGED (v4): 'no_gender' or 'gender'


if (
    (ver == 'v1' and heard_lang not in ['English', 'Urdu']) or
    (ver == 'v3' and heard_lang not in ['English', 'German']) or
    (ver == 'v4' and heard_lang not in ['English', 'Bangla'])                  # CHANGED (v4)
):
    raise ValueError(f"Invalid combination: ver={ver} and heard_lang={heard_lang}")


# assert ver in ['v1', 'v2', 'v3'], f"Invalid value for ver: {ver}"                                          # OLD
assert ver in ['v1', 'v2', 'v3', 'v4'], f"Invalid value for ver: {ver}"                                      # CHANGED (v4)
# assert heard_lang in ['English', 'Urdu', 'German'], f"Invalid value for heard_lang: {heard_lang}"           # OLD
assert heard_lang in ['English', 'Urdu', 'German', 'Bangla'], f"Invalid value for heard_lang: {heard_lang}"   # CHANGED (v4)

if ver == 'v1':
    assert heard_lang in ['English', 'Urdu'], f"Invalid combination: v1 can't be paired with {heard_lang}"
    unheard_lang = 'Urdu' if heard_lang == 'English' else 'English'

elif ver == 'v3':
    assert heard_lang in ['English', 'German'], f"Invalid combination: v3 can't be paired with {heard_lang}"
    unheard_lang = 'German' if heard_lang == 'English' else 'English'

elif ver == 'v4':                                                              # CHANGED (v4)
    assert heard_lang in ['English', 'Bangla'], f"Invalid combination: v4 can't be paired with {heard_lang}"
    unheard_lang = 'Bangla' if heard_lang == 'English' else 'English'

print('Heard_Language: %s'%(heard_lang))
print('Unheard Language: %s'%(unheard_lang))
if ver == 'v4':                                                                # CHANGED (v4)
    print('Track: %s'%(track))


# change the file path and dimensions (512) for face 
def read_data(ver, test_file_face, test_file_voice):
    print('Reading Test Face')
    
    face_test = pd.read_csv(test_file_face, header=None)
    print('Reading Test Voice')
    voice_test = pd.read_csv(test_file_voice, header=None)
    
    face_test = np.asarray(face_test)
    # face_test = face_test[:, :409]                                           # OLD (typo: truncated to 409)
    face_test = face_test[:, :4096]                                            # CHANGED: VGGFace is 4096-D
    voice_test = np.asarray(voice_test)
    # voice_test = voice_test[:, :512]                                         # OLD (VGGVox)
    voice_test = voice_test[:, :512 if ver != 'v4' else 192]                   # CHANGED (v4): ECAPA is 192-D
    
    face_test = torch.from_numpy(face_test).float()
    voice_test = torch.from_numpy(voice_test).float()
    print('  face %s   voice %s'%(face_test.shape, voice_test.shape))          # CHANGED (v4): shape check
    return face_test, voice_test

def test(face_test_heard, voice_test_heard, face_test_unheard, voice_test_unheard):
    # n_class = 64 if ver == 'v1' else 78 if ver == 'v2' else 50               # OLD
    n_class = 64 if ver == 'v1' else 78 if ver == 'v2' else 70 if ver == 'v4' else 50   # CHANGED (v4): 70 train speakers
    model = FOP(FLAGS, face_test_heard.shape[1], voice_test_heard.shape[1], n_class)
    # checkpoint = torch.load(FLAGS.ckpt)                                      # OLD
    checkpoint = torch.load(FLAGS.ckpt, weights_only=False,                    # CHANGED: torch 2.6 default
                            map_location='cuda' if FLAGS.cuda else 'cpu')      # CHANGED: load onto the available device
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})"
          .format('checkpoint.pth.tar', checkpoint['epoch']))
    model.eval()
    # model.cuda()                                                             # OLD: unconditional, crashes with no GPU
    if FLAGS.cuda:                                                             # CHANGED: GPU only when available
        model.cuda()
    print('  device: %s'%('cuda' if FLAGS.cuda else 'cpu'))                    # CHANGED (v4)
    
    if FLAGS.cuda:
        face_test_heard, voice_test_heard = face_test_heard.cuda(), voice_test_heard.cuda()
        face_test_unheard, voice_test_unheard = face_test_unheard.cuda(), voice_test_unheard.cuda()

    face_test_heard, voice_test_heard = Variable(face_test_heard), Variable(voice_test_heard)
    face_test_unheard, voice_test_unheard = Variable(face_test_unheard), Variable(voice_test_unheard)
    print('Computing scores')
    with torch.no_grad():
        _, face_heard, voice_heard = model(face_test_heard, voice_test_heard)
        _, face_unheard, voice_unheard = model(face_test_unheard, voice_test_unheard)
                
        face_heard, voice_heard = face_heard.data, voice_heard.data
        face_unheard, voice_unheard = face_unheard.data, voice_unheard.data
        
        face_heard, voice_heard = face_heard.cpu().detach().numpy(), voice_heard.cpu().detach().numpy()
        face_unheard, voice_unheard = face_unheard.cpu().detach().numpy(), voice_unheard.cpu().detach().numpy()
        
        scores_heard = np.linalg.norm(face_heard - voice_heard, axis=1, keepdims=True)
        scores_unheard = np.linalg.norm(face_unheard - voice_unheard, axis=1, keepdims=True)
        
        print('Writing scores to file')
        
        keys_heard = []
        keys_unheard = []

        V4_SPLIT = '/home/swapnilkhandoker/Bachelor_Thesis/project/utils/stage6/split_v1x/test'   # CHANGED (v4)
        key_heard_path = ('%s/%s/%s_test.txt'%(V4_SPLIT, track, heard_lang) if ver == 'v4'        # CHANGED (v4)
                          else './face_voice_association_splits/%s/%s_test.txt'%(ver, heard_lang))
        key_unheard_path = ('%s/%s/%s_test.txt'%(V4_SPLIT, track, unheard_lang) if ver == 'v4'    # CHANGED (v4)
                            else './face_voice_association_splits/%s/%s_test.txt'%(ver, unheard_lang))

        # with open('./face_voice_association_splits/%s/%s_test.txt'%(ver, heard_lang), 'r+') as f:      # OLD
        with open(key_heard_path, 'r+') as f:                                  # CHANGED
            for dat in f:
                if not dat.strip():                                            # CHANGED (v4): skip blank tail line
                    continue
                keys_heard.append(dat.split(' ')[0])
                
        # with open('./face_voice_association_splits/%s/%s_test.txt'%(ver, unheard_lang), 'r+') as f:    # OLD
        with open(key_unheard_path, 'r+') as f:                                # CHANGED
            for dat in f:
                if not dat.strip():                                            # CHANGED (v4): skip blank tail line
                    continue
                keys_unheard.append(dat.split(' ')[0])

        # CHANGED (v4): keys and feature rows must line up 1:1 (per-pair-line row contract)
        assert len(keys_heard) == len(scores_heard), \
            'heard: %d keys vs %d scores'%(len(keys_heard), len(scores_heard))
        assert len(keys_unheard) == len(scores_unheard), \
            'unheard: %d keys vs %d scores'%(len(keys_unheard), len(scores_unheard))
        
        with open('sub_score_%s_%s_heard.txt'%(ver, heard_lang), 'w') as f:
            for i, dat in enumerate(scores_heard):
                f.write('%s %f\n'%(keys_heard[i], dat))
                
        with open('sub_score_%s_%s_unheard.txt'%(ver, unheard_lang), 'w') as f:
            for i, dat in enumerate(scores_unheard):
                f.write('%s %f\n'%(keys_unheard[i], dat))

        print('  wrote sub_score_%s_%s_heard.txt   (%d lines)'%(ver, heard_lang, len(keys_heard)))      # CHANGED (v4)
        print('  wrote sub_score_%s_%s_unheard.txt (%d lines)'%(ver, unheard_lang, len(keys_unheard)))  # CHANGED (v4)
        
    return 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--cuda', action='store_true', default=False, help='CUDA training')
    parser.add_argument('--ckpt', type=str, default='./%s_models/%s_fop_model/checkpoint.pth.tar'%(ver, heard_lang), help='Checkpoints directory.')
    
    parser.add_argument('--dim_embed', type=int, default=128,
                        help='Embedding Size')
    parser.add_argument('--fusion', type=str, default='gated', help='Fusion Type')
    
    global FLAGS
    FLAGS, unparsed = parser.parse_known_args()
    FLAGS.cuda = torch.cuda.is_available()
    torch.manual_seed(FLAGS.seed)
    if FLAGS.cuda:
        torch.cuda.manual_seed(FLAGS.seed)

    V4_FEAT = '/home/swapnilkhandoker/Bachelor_Thesis/project/utils/stage7_feature_extraction/mavceleb_baseline/features_v1x'   # CHANGED (v4)

    print('Loading Heard Language Data')
    if ver == 'v4':                                                            # CHANGED (v4)
        test_file_face  = '%s/faces/test/%s/%s_test_faces.csv'%(V4_FEAT, track, heard_lang)
        test_file_voice = '%s/voices/test/%s/%s_test_voices.csv'%(V4_FEAT, track, heard_lang)
    else:
        test_file_face = './preExtracted_vggFace_utteranceLevel_Features/%s/%s/%s_faces_test.csv'%(ver, heard_lang, heard_lang)
        test_file_voice = './preExtracted_vggFace_utteranceLevel_Features/%s/%s/%s_voices_test.csv'%(ver, heard_lang, heard_lang)
    face_test_heard, voice_test_heard = read_data(ver, test_file_face, test_file_voice)

    print('Loading UnHeard Language Data')
    if ver == 'v4':                                                            # CHANGED (v4)
        test_file_face  = '%s/faces/test/%s/%s_test_faces.csv'%(V4_FEAT, track, unheard_lang)
        test_file_voice = '%s/voices/test/%s/%s_test_voices.csv'%(V4_FEAT, track, unheard_lang)
    else:
        test_file_face = './preExtracted_vggFace_utteranceLevel_Features/%s/%s/%s_faces_unheard_test.csv'%(ver, heard_lang, unheard_lang)
        test_file_voice = './preExtracted_vggFace_utteranceLevel_Features/%s/%s/%s_voices_unheard_test.csv'%(ver, heard_lang, unheard_lang)
    face_test_unheard, voice_test_unheard = read_data(ver, test_file_face, test_file_voice)

    test(face_test_heard, voice_test_heard, face_test_unheard, voice_test_unheard)