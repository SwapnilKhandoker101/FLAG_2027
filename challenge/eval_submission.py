
# This code will evaluate the performance based on scores submitted by participants



import numpy as np
from sklearn import metrics
from scipy.optimize import brentq
from scipy.interpolate import interp1d

# sub_ver = 'v1'                        # OLD
# test_lang = 'English'                 # OLD
sub_ver = 'v4'                          # CHANGED (v4)
test_lang = 'Bangla'                   # CHANGED: language being TESTED ('English' or 'Bangla')
track = 'no_gender'                     # CHANGED (v4): 'no_gender' or 'gender'
cond = 'unheard'                          # CHANGED (v4): 'heard' or 'unheard'
                                        #   heard   -> model trained on test_lang
                                        #   unheard -> model trained on the other language
SCORE_ROOT = '/home/swapnilkhandoker/Bachelor_Thesis/project/utils/stage7_feature_extraction/mavceleb_baseline/sub_score_v4'   # CHANGED (v4)

# Read scores submitted by participant 'participantID_version_trainlang_testlang.txt'
sub_results = {}
# with open('./template_submission/sub_score_%s.txt'%(test_lang), 'r') as f:                     # OLD
sub_path = '%s/%s/sub_score_v4_%s_%s.txt'%(SCORE_ROOT, track, test_lang, cond)                   # CHANGED (v4)
print('scores: %s'%sub_path)                                                                     # CHANGED (v4)
with open(sub_path, 'r') as f:                                                                   # CHANGED (v4)
    for dat in f:
        if not dat.strip():                                                                      # CHANGED (v4): skip blank tail line
            continue
        key, score = dat.rstrip('\n').split(' ')
        sub_results[key] = score


ref_keys = []
gt = []
# GT read from local file for submitted version and test language
# with open('./ground_truth/%s_test.txt'%(test_lang), 'r') as f:                                 # OLD
gt_path = './ground_truth/%s_test_%s.txt'%(test_lang, track)                                     # CHANGED (v4)
print('truth : %s'%gt_path)                                                                      # CHANGED (v4)
with open(gt_path, 'r') as f:                                                                    # CHANGED (v4)
    for i, dat in enumerate(f):
        if not dat.strip():                                                                      # CHANGED (v4): skip blank tail line
            continue
        tmp = dat.split(' ')
        ref_keys.append(tmp[0])
        gt.append(int(tmp[1]))


# Arrange submission scores to the same pattern as ground truth keys
sub_scores = []
sub_keys = []
for key in ref_keys:
    sub_scores.append(float(sub_results[key]))
    sub_keys.append(key)
assert sub_keys == ref_keys


score = np.asarray(sub_scores)
gt = np.asarray(gt)

fpr, tpr, thresholds = metrics.roc_curve(gt, -score)
fnr = 1-tpr
# eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
print(eer)
# print(round(eer*100, 1))                                                                       # OLD
print('%s / %s / %s   pairs=%d  pos=%d  EER = %.1f'                                              # CHANGED (v4)
      %(test_lang, track, cond, len(gt), int(gt.sum()), eer*100))