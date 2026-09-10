#!/usr/bin/env python3
"""
extract_face_features_v3.py — VGGFace (4096-D) features for split_v3.

Same encoder, same weights, same preprocessing as extract_face_features.py.
Only the input layout and output naming changed.

What changed from the general_split version
-------------------------------------------
1. BUCKET DISCOVERY. The old script looked for <split>/pairs_<track>.txt and
   ran train/valid/test for ONE track per invocation. split_v3 has ten lists:

       train/train_English.txt          train/train_Bangla.txt
       valid/gender/English_valid.txt   valid/gender/Bangla_valid.txt
       valid/no_gender/...              test/gender/...   test/no_gender/...

   All ten are found and processed in one run.

2. TRAIN LABELS COME FROM THE LIST. The old script rebuilt the label by
   sorting speaker ids and enumerating them, in both the face and the voice
   extractor independently -- two chances to disagree. Train lines now carry
   the class index as field 6, written by build_split_v3.py alongside
   label_map_<lang>.csv, so both extractors read the same authoritative number.

3. ONE OUTPUT CSV PER BUCKET, mirroring the split tree:
       <out>/train/train_English_faces.csv        4096 cols + label
       <out>/test/gender/English_test_faces.csv   4096 cols, no label
   plus <bucket>_faces_rowmap.csv recording row -> source path, which is what
   lets a later script prove the row contract instead of assuming it.

UNCHANGED and deliberately so
-----------------------------
* create_model() and the Flatten fix on layers[-3]
* load_img(): BGR conversion and the VGGFace channel means
* per-pair-line row contract -- row i is line i, no deduplication in the output
* embedding cache so each distinct image is encoded once
* failed loads written as TRUE ZERO VECTORS, never as an embedded black image

Run
---
    conda activate vggface
    python extract_face_features_v3.py \
        --split_root /home/.../stage6/split_v3 \
        --src_root   /home/.../stage6/general_split \
        --weights    models/vgg_face_weights.h5 \
        --out_dir    features_v3/faces --batch_size 16

Extract from split_v3 (real paths), not release_v3: the cache is keyed on path,
and in release_v3 every duplicate row is a separate file, so the same image
would be encoded several times for identical output.
"""

import os
import argparse
import csv

import numpy as np
from tqdm import tqdm
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (ZeroPadding2D, Convolution2D,
                                     MaxPooling2D, Dropout, Flatten)
from PIL import Image

FACE_DIM = 4096


# ---------------------------------------------------------------------------
# Model — unchanged
# ---------------------------------------------------------------------------

def create_model(weights_path):
    model = Sequential()
    model.add(ZeroPadding2D((1,1), input_shape=(224,224,3)))
    model.add(Convolution2D(64,(3,3),activation='relu')); model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64,(3,3),activation='relu')); model.add(MaxPooling2D((2,2),strides=(2,2)))
    model.add(ZeroPadding2D((1,1))); model.add(Convolution2D(128,(3,3),activation='relu')); model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128,(3,3),activation='relu')); model.add(MaxPooling2D((2,2),strides=(2,2)))
    model.add(ZeroPadding2D((1,1))); model.add(Convolution2D(256,(3,3),activation='relu')); model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256,(3,3),activation='relu')); model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256,(3,3),activation='relu')); model.add(MaxPooling2D((2,2),strides=(2,2)))
    model.add(ZeroPadding2D((1,1))); model.add(Convolution2D(512,(3,3),activation='relu')); model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512,(3,3),activation='relu')); model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512,(3,3),activation='relu')); model.add(MaxPooling2D((2,2),strides=(2,2)))
    model.add(ZeroPadding2D((1,1))); model.add(Convolution2D(512,(3,3),activation='relu')); model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512,(3,3),activation='relu')); model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512,(3,3),activation='relu')); model.add(MaxPooling2D((2,2),strides=(2,2)))
    model.add(Convolution2D(4096,(7,7),activation='relu')); model.add(Dropout(0.5))   # fc6
    model.add(Convolution2D(4096,(1,1),activation='relu')); model.add(Dropout(0.5))   # fc7
    model.add(Convolution2D(2622,(1,1))); model.add(Flatten())
    model.load_weights(weights_path)

    # layers[-3] is the fc7 Dropout, output (B, 1, 1, 4096) -- must be flattened
    feat = Flatten()(model.layers[-3].output)
    out = Model(inputs=model.layers[0].input, outputs=feat)

    shape = tuple(out.output_shape)
    if len(shape) != 2 or shape[1] != FACE_DIM:
        raise SystemExit('Expected model output (None, %d), got %s' % (FACE_DIM, shape))
    print('Model ready, output shape %s' % (shape,))
    return out


def load_img(path):
    im = Image.open(path).convert('RGB').resize((224, 224))
    arr = np.asarray(im, dtype=np.float32)          # (224,224,3) RGB 0-255
    arr = arr[..., ::-1].copy()                     # RGB -> BGR (VGGFace)
    arr[..., 0] -= 93.5940
    arr[..., 1] -= 104.7624
    arr[..., 2] -= 129.1863
    return arr


def fmt(vec):
    return ','.join('%.6g' % v for v in vec)


# ---------------------------------------------------------------------------
# split_v3 layout
# ---------------------------------------------------------------------------

def find_buckets(split_root):
    """(rel_dir, bucket, txt_path, kind) for every list under split_root."""
    out = []
    for dirpath, _, filenames in os.walk(split_root):
        rel = os.path.relpath(dirpath, split_root)
        rel = '' if rel == '.' else rel
        for fn in sorted(filenames):
            if fn.endswith('.txt'):
                bucket = fn[:-4]
                kind = 'train' if bucket.startswith('train_') else 'eval'
                out.append((rel, bucket, os.path.join(dirpath, fn), kind))
    return sorted(out)


def read_list(path, kind):
    """
    One dict per line, in file order. Train lines carry two extra fields:
        pair_id label voice face speaker_id class_idx
    """
    rows = []
    with open(path) as f:
        for ln, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            p = line.split()
            need = 6 if kind == 'train' else 4
            if len(p) < need:
                raise SystemExit('%s line %d: expected %d fields, got %d'
                                 % (path, ln + 1, need, len(p)))
            r = {'pair_id': p[0], 'label': p[1], 'voice': p[2], 'face': p[3]}
            if kind == 'train':
                r['speaker'] = p[4]
                r['cls'] = int(p[5])
            rows.append(r)
    return rows


# ---------------------------------------------------------------------------
# Embedding — unchanged
# ---------------------------------------------------------------------------

def embed_unique(model, paths, src_root, batch_size, desc):
    """
    Embed each DISTINCT image once. A failed load maps to None and is written
    as a true zero vector by the caller, never as an embedded black image.
    """
    cache, failed = {}, 0
    batch_paths, batch_imgs = [], []

    def flush():
        if not batch_imgs:
            return
        feats = model.predict(np.stack(batch_imgs), verbose=0)
        feats = feats.reshape(feats.shape[0], -1)
        if feats.shape[1] != FACE_DIM:
            raise SystemExit('Got %d features, expected %d'
                             % (feats.shape[1], FACE_DIM))
        for p, fv in zip(batch_paths, feats):
            cache[p] = fv.astype(np.float32)
        batch_paths.clear(); batch_imgs.clear()

    for p in tqdm(paths, desc=desc, ncols=80):
        try:
            batch_imgs.append(load_img(os.path.join(src_root, p)))
            batch_paths.append(p)
        except Exception as e:
            tqdm.write('    [FAIL] %s: %s' % (p, e))
            cache[p] = None
            failed += 1
            continue
        if len(batch_imgs) >= batch_size:
            flush()
    flush()
    return cache, failed


# ---------------------------------------------------------------------------

def extract_bucket(model, rel, bucket, txt_path, kind, src_root, out_dir,
                   batch_size):
    items = read_list(txt_path, kind)
    with_label = (kind == 'train')

    if with_label:
        n_neg = sum(1 for r in items if r['label'] != '1')
        if n_neg:
            print('    *** %d of %d train rows are NOT label 1. FOP trains a '
                  'speaker classifier; those rows are mislabelled by '
                  'construction.' % (n_neg, len(items)))
        n_cls = len({r['cls'] for r in items})
        print('    %d classes (from the list, not re-derived)' % n_cls)

    uniq = sorted({r['face'] for r in items})
    label = ('%s/%s' % (rel, bucket)) if rel else bucket
    print('  %s: %d rows, %d unique images' % (label, len(items), len(uniq)))

    cache, failed = embed_unique(model, uniq, src_root, batch_size, label)

    dest = os.path.join(out_dir, rel) if rel else out_dir
    os.makedirs(dest, exist_ok=True)
    out_csv = os.path.join(dest, '%s_faces.csv' % bucket)
    out_map = os.path.join(dest, '%s_faces_rowmap.csv' % bucket)

    n_bad = 0
    with open(out_csv, 'w', newline='') as fout:
        for r in items:
            fv = cache.get(r['face'])
            if fv is None:
                fv = np.zeros(FACE_DIM, dtype=np.float32)
                n_bad += 1
            line = fmt(fv)
            if with_label:
                line += ',%d' % r['cls']
            fout.write(line + '\n')

    with open(out_map, 'w', newline='') as fmap:
        w = csv.writer(fmap, lineterminator='\n')
        cols = ['row', 'pair_id', 'label', 'face_path', 'voice_path']
        if with_label:
            cols += ['speaker_id', 'class_idx']
        w.writerow(cols)
        for i, r in enumerate(items):
            row = [i, r['pair_id'], r['label'], r['face'], r['voice']]
            if with_label:
                row += [r['speaker'], r['cls']]
            w.writerow(row)

    print('  wrote %s  (%d rows x %d dims%s)'
          % (out_csv, len(items), FACE_DIM, ' + label' if with_label else ''))
    print('  wrote %s' % out_map)
    if failed:
        print('    *** %d image(s) failed to load, affecting %d row(s), written '
              'as zero vectors. Check before training.' % (failed, n_bad))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split_root', required=True, help='split_v3 from Stage 6')
    ap.add_argument('--src_root', required=True,
                    help='dir the list paths are relative to (general_split)')
    ap.add_argument('--weights', default='models/vgg_face_weights.h5')
    ap.add_argument('--out_dir', default='features_v3/faces')
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--only', help='substring filter, e.g. English_valid')
    args = ap.parse_args()

    buckets = find_buckets(args.split_root)
    if args.only:
        buckets = [b for b in buckets if args.only in b[1]]
    if not buckets:
        raise SystemExit('No lists found under %s' % args.split_root)
    print('Found %d bucket(s):' % len(buckets))
    for rel, bucket, _, kind in buckets:
        print('   %-22s %s' % (('%s/%s' % (rel, bucket)) if rel else bucket, kind))

    print('\nLoading VGGFace...')
    model = create_model(args.weights)

    for rel, bucket, txt_path, kind in buckets:
        extract_bucket(model, rel, bucket, txt_path, kind, args.src_root,
                       args.out_dir, args.batch_size)
    print('Done.')


if __name__ == '__main__':
    main()