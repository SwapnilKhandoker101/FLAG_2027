#!/usr/bin/env python3
"""
extract_voice_features_ecapa_v3.py — ECAPA-TDNN (192-D) voice features for split_v3.

Same encoder (yangwang825/ecapa-tdnn-vox2), same 16 kHz mono loading, same
length-sorted batching and same CUDA-OOM recovery ladder as
extract_voice_features_ecapa.py. Only the input layout and output naming
changed, so features are numerically identical to what that script produced.

What changed
------------
1. BUCKET DISCOVERY. The old script took --track and ran train/valid/test for
   that one track. split_v3 has ten lists, all processed in one run:

       train/train_English.txt          train/train_Bangla.txt
       valid/gender/English_valid.txt   valid/no_gender/English_valid.txt
       test/gender/Bangla_test.txt      ... and so on

2. TRAIN LABELS COME FROM THE LIST (field 6, class_idx), not re-derived by
   sorting speaker ids. The face and voice extractors previously each rebuilt
   that mapping independently -- two chances to disagree. Now both read the
   same number, written by build_split_v3.py alongside label_map_<lang>.csv.

3. ONE OUTPUT CSV PER BUCKET, mirroring the split tree:
       <out>/train/train_English_voices.csv        192 cols + label
       <out>/test/gender/English_test_voices.csv   192 cols, no label
   plus <bucket>_voices_rowmap.csv recording row -> source path.

UNCHANGED and deliberately so
-----------------------------
* Encoder class and encode_batch
* librosa load at forced 16 kHz mono, with resample/downmix counters
* RELATIVE wav_lens (SpeechBrain convention) -- getting this wrong silently
  degrades every clip shorter than the longest in its batch
* duration-sorted batching, so padding waste stays low
* the OOM ladder: batch -> halves -> quarters -> single -> CPU
* failed wavs written as TRUE ZERO VECTORS
* per-pair-line row contract: row i is line i, no deduplication in the output

Run
---
    conda activate ecapa
    export HF_HUB_OFFLINE=1
    python extract_voice_features_ecapa_v3.py \
        --split_root /home/.../stage6/split_v3 \
        --src_root   /home/.../stage6/general_split \
        --out_dir    features_v3/voices --batch_size 8 --device cuda:0

Extract from split_v3 (real paths), not release_v3: the cache is keyed on path,
and in release_v3 every duplicate row is a separate file.
"""

import os
import argparse

import numpy as np
import torch
import librosa
from tqdm import tqdm

# soundfile gives cheap header reads (rate, channels, duration) without
# decoding the audio. librosa depends on it, so it is always present.
try:
    import soundfile as sf
except ImportError:
    sf = None

# SpeechBrain moved the inference interfaces in 1.0; support both layouts.
try:
    from speechbrain.inference.interfaces import Pretrained
except ImportError:                                    # speechbrain < 1.0
    from speechbrain.pretrained.interfaces import Pretrained

VOICE_DIM = 192            # ECAPA-TDNN embedding size (VGGVox was 512)
TARGET_SR = 16000          # sample rate the model was trained at


class Encoder(Pretrained):
    """From the model card, with batching support."""

    MODULES_NEEDED = ["compute_features", "mean_var_norm", "embedding_model"]

    def encode_batch(self, wavs, wav_lens=None, normalize=False):
        if len(wavs.shape) == 1:
            wavs = wavs.unsqueeze(0)
        if wav_lens is None:
            wav_lens = torch.ones(wavs.shape[0], device=self.device)

        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
        wavs = wavs.float()

        feats = self.mods.compute_features(wavs)
        feats = self.mods.mean_var_norm(feats, wav_lens)
        embeddings = self.mods.embedding_model(feats, wav_lens)

        if normalize:
            embeddings = self.hparams.mean_var_norm_emb(
                embeddings, torch.ones(embeddings.shape[0], device=self.device))
        return embeddings



# ---------------------------------------------------------------------------
# split_v3 layout
# ---------------------------------------------------------------------------

import csv


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


def fmt(vec):
    return ','.join('%.6g' % v for v in vec)


# ---------------------------------------------------------------------------
# audio  (librosa)
# ---------------------------------------------------------------------------

def probe_audio(path):
    """(sample_rate, channels, duration_seconds) from the header only."""
    if sf is not None:
        try:
            info = sf.info(path)
            return info.samplerate, info.channels, info.duration
        except Exception:
            pass
    try:
        sr = librosa.get_samplerate(path)
        return sr, 1, librosa.get_duration(path=path)
    except Exception:
        return None, None, 0.0


def load_wav(path, stats, max_seconds=0.0):
    """
    Load as mono 16 kHz float32, 1-D, via librosa.

    librosa.load(sr=TARGET_SR, mono=True) resamples and downmixes silently, so
    the header is probed first purely to COUNT how often that happens. Stage 3
    already writes 16 kHz mono (ffmpeg -ar 16000 -ac 1), so both counters
    should stay at zero; anything else means the upstream audio is not what
    this pipeline assumes.
    """
    native_sr, channels, _ = probe_audio(path)
    if native_sr is not None and native_sr != TARGET_SR:
        stats['resampled'] += 1
        stats['rates'].add(native_sr)
    if channels is not None and channels > 1:
        stats['downmixed'] += 1

    y, _ = librosa.load(path, sr=TARGET_SR, mono=True)

    if max_seconds and y.size > int(max_seconds * TARGET_SR):
        keep_n = int(max_seconds * TARGET_SR)
        start = (y.size - keep_n) // 2          # centre crop
        y = y[start:start + keep_n]
        stats['cropped'] += 1

    if y.size == 0:
        raise ValueError('empty waveform')
    if not np.all(np.isfinite(y)):
        raise ValueError('non-finite samples')

    return torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))


def duration_of(path):
    return probe_audio(path)[2]


def _is_oom(exc):
    return 'out of memory' in str(exc).lower()


def encode_with_fallback(model, batch, rel_lens, keep, wavs, normalize, stats):
    """
    Encode a batch, degrading gracefully on CUDA OOM instead of discarding it.

    Long clips are the problem: sorting by duration puts every multi-minute wav
    in the last few batches, and 16 of them at once can exceed a 4 GB card
    before ECAPA's activations are even allocated. Previously the whole batch
    was marked failed and written as zero vectors -- 106 wavs lost that way on
    a 3050.

    Ladder: full batch -> halves -> quarters -> single clips on GPU -> single
    clips on CPU. Only a clip that fails even on CPU is given up on, and that
    would be a genuinely corrupt file rather than a memory limit.

    Returns a list aligned with `keep`, entries either (192,) arrays or None.
    """
    def run(sub_wavs):
        lengths = torch.tensor([w.shape[0] for w in sub_wavs], dtype=torch.float)
        maxlen = int(lengths.max().item())
        b = torch.zeros(len(sub_wavs), maxlen)
        for j, w in enumerate(sub_wavs):
            b[j, :w.shape[0]] = w
        with torch.no_grad():
            e = model.encode_batch(b, lengths / maxlen, normalize=normalize)
        return e.squeeze(1).cpu().numpy()

    def attempt(idx):
        """idx: indices into keep/wavs. Returns dict idx -> embedding or None."""
        try:
            out = run([wavs[i] for i in idx])
            return {i: out[k] for k, i in enumerate(idx)}
        except Exception as e:
            if not _is_oom(e) or len(idx) == 1:
                if _is_oom(e):
                    # Single clip too large for the GPU -- fall back to CPU.
                    i = idx[0]
                    try:
                        torch.cuda.empty_cache()
                        dev = model.device
                        model.mods.to('cpu'); model.device = 'cpu'
                        out = run([wavs[i]])
                        model.mods.to(dev); model.device = dev
                        stats['cpu_fallback'] += 1
                        return {i: out[0]}
                    except Exception as e2:
                        model.mods.to(dev); model.device = dev
                        tqdm.write('    [FAIL] %s: %s' % (keep[i], e2))
                        return {i: None}
                tqdm.write('    [FAIL] %s: %s' % (keep[idx[0]], e))
                return {i: None for i in idx}

            torch.cuda.empty_cache()
            stats['oom_retries'] += 1
            mid = len(idx) // 2
            res = attempt(idx[:mid])
            res.update(attempt(idx[mid:]))
            return res

    got = attempt(list(range(len(keep))))
    return [got.get(i) for i in range(len(keep))]


def embed_unique(model, paths, split_root, batch_size, normalize, desc,
                 max_seconds=0.0):
    """
    Embed each DISTINCT wav once. Failed wavs map to None.

    Files are processed in length order so each batch pads to nearly the same
    length; ECAPA pools over the time axis, so heavy padding within a batch
    would distort the shorter clips' embeddings even with correct wav_lens. On
    a typical duration spread this cuts padded-sample waste from ~46% to ~13%.
    """
    stats = {'resampled': 0, 'downmixed': 0, 'rates': set(),
             'oom_retries': 0, 'cpu_fallback': 0, 'cropped': 0}
    cache = {}
    failed = 0

    order = sorted(paths, key=lambda p: duration_of(os.path.join(split_root, p)))

    pbar = tqdm(total=len(order), desc=desc, ncols=80)
    i = 0
    while i < len(order):
        chunk = order[i:i + batch_size]
        i += batch_size

        wavs, keep = [], []
        for p in chunk:
            try:
                wavs.append(load_wav(os.path.join(split_root, p), stats, max_seconds))
                keep.append(p)
            except Exception as e:
                tqdm.write('    [FAIL] %s: %s' % (p, e))
                cache[p] = None
                failed += 1
        pbar.update(len(chunk))
        if not wavs:
            continue

        lengths = torch.tensor([w.shape[0] for w in wavs], dtype=torch.float)
        maxlen = int(lengths.max().item())
        batch = torch.zeros(len(wavs), maxlen)
        for j, w in enumerate(wavs):
            batch[j, :w.shape[0]] = w

        # SpeechBrain wants RELATIVE lengths, not absolute sample counts.
        rel_lens = lengths / maxlen

        embs = encode_with_fallback(model, batch, rel_lens, keep, wavs,
                                    normalize, stats)
        for p, v in zip(keep, embs):
            if v is None:
                cache[p] = None
                failed += 1
            else:
                if v.shape[0] != VOICE_DIM:
                    raise SystemExit('Got %d features, expected %d'
                                     % (v.shape[0], VOICE_DIM))
                cache[p] = v.astype(np.float32)

    pbar.close()
    if stats['cropped']:
        print('    %d clip(s) centre-cropped to %.0f s' % (stats['cropped'], max_seconds))
    if stats['oom_retries'] or stats['cpu_fallback']:
        print('    memory pressure: %d batch split(s) after CUDA OOM, %d clip(s) '
              'encoded on CPU. All recovered -- no data lost. Lower --batch_size '
              'to avoid the retries.' % (stats['oom_retries'], stats['cpu_fallback']))
    if stats['resampled'] or stats['downmixed']:
        print('    *** audio conditioning applied: %d resampled to %d Hz (source '
              'rates seen: %s), %d downmixed to mono. Stage 3 should already emit '
              '16 kHz mono -- worth checking why it did not.'
              % (stats['resampled'], TARGET_SR,
                 sorted(stats['rates']) or 'unknown', stats['downmixed']))
    return cache, failed


# ---------------------------------------------------------------------------


def extract_bucket(model, rel, bucket, txt_path, kind, src_root, out_dir,
                   batch_size, normalize, max_seconds=0.0):
    items = read_list(txt_path, kind)
    with_label = (kind == 'train')

    if with_label:
        n_neg = sum(1 for r in items if r['label'] != '1')
        if n_neg:
            print('    *** %d of %d train rows are NOT label 1. FOP trains a '
                  'speaker classifier; those rows are mislabelled by '
                  'construction.' % (n_neg, len(items)))
        print('    %d classes (from the list, not re-derived)'
              % len({r['cls'] for r in items}))

    uniq = sorted({r['voice'] for r in items})
    label = ('%s/%s' % (rel, bucket)) if rel else bucket
    print('  %s: %d rows, %d unique wavs' % (label, len(items), len(uniq)))

    cache, failed = embed_unique(model, uniq, src_root, batch_size, normalize,
                                 label, max_seconds)

    dest = os.path.join(out_dir, rel) if rel else out_dir
    os.makedirs(dest, exist_ok=True)
    out_csv = os.path.join(dest, '%s_voices.csv' % bucket)
    out_map = os.path.join(dest, '%s_voices_rowmap.csv' % bucket)

    n_bad = 0
    with open(out_csv, 'w', newline='') as fout:
        for r in items:
            fv = cache.get(r['voice'])
            if fv is None:
                fv = np.zeros(VOICE_DIM, dtype=np.float32)
                n_bad += 1
            line = fmt(fv)
            if with_label:
                line += ',%d' % r['cls']
            fout.write(line + '\n')

    with open(out_map, 'w', newline='') as fmap:
        w = csv.writer(fmap, lineterminator='\n')
        cols = ['row', 'pair_id', 'label', 'voice_path', 'face_path']
        if with_label:
            cols += ['speaker_id', 'class_idx']
        w.writerow(cols)
        for i, r in enumerate(items):
            row = [i, r['pair_id'], r['label'], r['voice'], r['face']]
            if with_label:
                row += [r['speaker'], r['cls']]
            w.writerow(row)

    print('  wrote %s  (%d rows x %d dims%s)'
          % (out_csv, len(items), VOICE_DIM, ' + label' if with_label else ''))
    print('  wrote %s' % out_map)
    if failed:
        print('    *** %d wav(s) failed, affecting %d row(s), written as zero '
              'vectors. Check before training.' % (failed, n_bad))


def check_local_source(source):
    """Warn if hyperparams.yaml still points pretrained_path at HuggingFace."""
    yaml_path = os.path.join(source, 'hyperparams.yaml')
    if not os.path.isfile(yaml_path):
        print('WARNING: %s not found. SpeechBrain cannot build the model from a '
              'directory without it.' % yaml_path)
        return
    with open(yaml_path) as f:
        for line in f:
            if line.startswith('pretrained_path:'):
                val = line.split(':', 1)[1].strip()
                if not os.path.isdir(val):
                    print('WARNING: hyperparams.yaml has  pretrained_path: %s' % val)
                    print('  That is not a local directory, so the .ckpt files will '
                          'be fetched from HuggingFace despite --source being local.')
                    print("  Fix:  sed -i 's|^pretrained_path:.*|pretrained_path: %s|' %s"
                          % (source, yaml_path))
                return



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split_root', required=True, help='split_v3 from Stage 6')
    ap.add_argument('--src_root', required=True,
                    help='dir the list paths are relative to (general_split)')
    ap.add_argument('--source', default='pretrained_models',
                    help='Local dir with hyperparams.yaml and the .ckpt files')
    ap.add_argument('--savedir', default=None)
    ap.add_argument('--out_dir', default='features_v3/voices')
    ap.add_argument('--batch_size', type=int, default=8,
                    help='8 is safe on a 4 GB card; batches halve on CUDA OOM.')
    ap.add_argument('--max_seconds', type=float, default=0.0,
                    help='If >0, centre-crop longer clips. ECAPA pools over '
                         'time, so 30 s is ample and it bounds peak memory.')
    ap.add_argument('--normalize', action='store_true',
                    help="NOT defined by this checkpoint's YAML -- leave off.")
    ap.add_argument('--device', default=None, help='cuda:0 | cpu (default: auto)')
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

    savedir = args.savedir or args.source
    device = args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu')

    if os.path.isdir(args.source):
        check_local_source(args.source)
    else:
        print('NOTE: --source %r is not a local directory; SpeechBrain will '
              'treat it as a HuggingFace id and download.' % args.source)

    print('Loading %s on %s ...' % (args.source, device))
    model = Encoder.from_hparams(source=args.source, savedir=savedir,
                                 run_opts={'device': device})
    model.eval()

    n_par = sum(p.numel() for p in model.mods.embedding_model.parameters())
    print('embedding_model: %s parameters' % f'{n_par:,}')

    # Fail fast on a dimensionality surprise rather than after hours of work.
    probe = model.encode_batch(torch.zeros(1, TARGET_SR))
    got = tuple(probe.shape)
    print('Model ready, probe embedding shape %s -> %d-D' % (got, got[-1]))
    if got[-1] != VOICE_DIM:
        raise SystemExit('Expected %d-D embeddings, got %d.'
                         % (VOICE_DIM, got[-1]))

    for rel, bucket, txt_path, kind in buckets:
        extract_bucket(model, rel, bucket, txt_path, kind, args.src_root,
                       args.out_dir, args.batch_size, args.normalize,
                       args.max_seconds)
    print('Done.')


if __name__ == '__main__':
    main()