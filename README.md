# FLAG 2027 Challenge — Baseline

**Face-voice Association across LAnguages and Gender (FLAG) 2027**
Accepted as an ICASSP 2027 Grand Challenge, supported by the IEEE Signal
Processing Society Challenge Program.



---

## Task

Face-voice association is established through a **cross-modal verification**
task: given a single sample containing both a face and a voice, verify whether
both belong to the same identity.

FLAG 2027 extends this along two dimensions, evaluated as separate tracks:

| Track | Description |
| --- | --- |
| **Language impact** | Whether a voice recorded in a different language is still correctly associated with the speaker's face. Evaluated on a *heard* language (seen in training) and an *unheard* language. |
| **Gender impact** | Gender-controlled verification — negative pairs are restricted to speakers of the same gender as the positive, removing gender as an identity shortcut. |

Train and test splits follow the **unseen–unheard** configuration: disjoint
speakers, disjoint languages.

## Dataset

Built on **MAV-Celeb v4**, extending the prior FAME 2024 and FAME 2026 splits
with a new set of **100 English–Bengali speakers**, annotated for language and
gender. Samples are drawn from YouTube interviews, talk shows, and television
debates, and include real-world variation in pose, lighting, motion blur,
occlusion, background chatter, and compression artifacts.

[Download]([https://drive.google.com/drive/folders/1OJyjXJULErvrvzLQmpJn5v8rRo0n_fod?usp=sharing](https://drive.google.com/drive/folders/1YFVLHIWu0yBQYOIgjTvfK7d_M_fsfzn8?usp=sharing))
```
data/
├── voices/
│   ├── English/
│   └── Bengali/
└── faces/
    ├── English/
    └── Bengali/
```

Pair list files use the format:

```
ysuvkz41  voices/English/00000.wav  faces/English/00000.jpg
tog3zj45  voices/English/00001.wav  faces/English/00001.jpg
```

## Baseline

The baseline is a two-branch network over pre-extracted face and voice
embeddings, trained with an orthogonality constraint on the multimodal
embeddings of different speakers. It follows *Fusion and Orthogonal Projection
for Improved Face-Voice Association*
([paper](https://ieeexplore.ieee.org/abstract/document/9747704) ·
[code](https://github.com/msaadsaeed/FOP)).

| Component | Model |
| --- | --- |
| Face encoder | VGGFace |
| Voice encoder | ECAPA-TDNN |
| Fusion | Two-branch network with orthogonal projection |

## Hierarchy

```
.
├── challenge/
│   └── eval_submission.py            # scores a submission file against ground truth
├── feature_extraction/
│   ├── backbone.py
│   ├── extract_face_features.py      # VGGFace embeddings from face crops
│   ├── extract_voice_features_ecapa.py
│   ├── model.py
│   └── utils.py
├── computeScore.py                   # EER computation
├── main.py                           # training entry point
├── online_evaluation.py              # evaluation during training
├── retrieval_model.py                # two-branch fusion model
└── test.py                           # inference / score-file generation
```


## Setup

We used Anaconda to set up the environment for our experiments:

```
python==3.10.0
```

[CUDA](https://developer.nvidia.com/cuda-toolkit-archive) and
[cuDNN](https://developer.nvidia.com/rdp/cudnn-archive) setup:

- CUDA Toolkit 12.4
- cuDNN v9.x for CUDA 12.x

To install PyTorch with GPU support:

```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

Remaining dependencies:

```bash
pip install -r requirements.txt

```
