"""
Improved 3D axis-angle extraction for the 30 already-2D-validated poses.

The original 2D pipeline (retarget.py) picks the lowest-motion frame per pose using
only a velocity-smoothness score, then flattens 3D joints into a single sagittal-plane
2D projection. That flattening hides left-right asymmetry (a 3D pose where one leg is
mid-transition while the other is planted can still project into a plausible-looking 2D
stick figure). When the SAME picked frame is used to drive a real 3D rig directly
(no flattening), that asymmetry becomes obvious and wrong (confirmed visually for
urdhva-dhanurasana: one leg's foot ended up far above the head instead of both feet
level on the ground).

This script re-selects, per pose, the frame that is BOTH low-motion AND (for
bilaterally-symmetric poses) left-right symmetric, then applies a small final
symmetrization pass (mirror-averaging matched joint pairs) to remove residual mocap
noise. Asymmetric (inherently one-sided) poses are left on the original 2D-pipeline's
chosen frame, just re-extracted as raw axis-angle instead of a 2D projection.

World frame: Z-up (confirmed empirically in retarget.py). Left-right mirror axis: X
(confirmed by dumping the T-pose bind skeleton: left-side joints have +X, right-side -X).
"""
import glob
import json
import os
import pickle

import numpy as np
import torch
import smplx

DATA_DIR = r"C:\SAVRO\moyo_toolkit_inspect\data\mosh_extracted\mosh"
MODEL_DIR = r"C:\SAVRO\smplx_models\standard\models"
DEBUG_JSON = r"C:\SAVRO\moyo_toolkit_inspect\retargeted_debug.json"
OUT = r"C:\SAVRO\moyo_toolkit_inspect\axis_angle_poses_v2.json"
OUT_APP = r"C:\SAVRO\Yoga App\axis_angle_poses.json"

JOINT_NAMES = ['pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee', 'spine2',
               'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot', 'neck',
               'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder',
               'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist']
J = {n: i for i, n in enumerate(JOINT_NAMES)}

# Left-right joint pairs among the 22 body joints (index into JOINT_NAMES / axis-angle triples).
LR_PAIRS = [(1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21)]

ASYMMETRIC = {
    'ardha-chandrasana', 'ardha-matsyendrasana', 'gomukhasana', 'natarajasana',
    'parsvottanasana', 'vasisthasana', 'virabhadrasana-3', 'vrksasana',
}
ACCEPTED = [
    'adho-mukha-svanasana', 'ananda-balasana', 'ardha-chandrasana', 'ardha-matsyendrasana',
    'ardha-pincha-mayurasana', 'bakasana', 'chaturanga-dandasana', 'dandasana', 'dhanurasana',
    'gomukhasana', 'halasana', 'malasana', 'marjaryasana-bitilasana', 'natarajasana', 'navasana',
    'parsvottanasana', 'phalakasana', 'purvottanasana', 'sarvangasana', 'setu-bandhasana',
    'supta-baddha-konasana', 'tolasana', 'urdhva-dhanurasana', 'ustrasana', 'utkatasana',
    'uttana-shishosana', 'vasisthasana', 'viparita-karani', 'virabhadrasana-3', 'vrksasana',
]

print("Loading SMPL-X model...")
model = smplx.create(MODEL_DIR, model_type='smplx', gender='female', use_pca=False,
                      flat_hand_mean=True, num_betas=10, ext='npz')


def smplx_joints(fullpose, trans):
    fp = torch.tensor(fullpose, dtype=torch.float32)
    tr = torch.tensor(trans, dtype=torch.float32)
    n = fp.shape[0]
    with torch.no_grad():
        out = model(
            global_orient=fp[:, :3], body_pose=fp[:, 3:66], jaw_pose=fp[:, 66:69],
            leye_pose=fp[:, 69:72], reye_pose=fp[:, 72:75],
            left_hand_pose=fp[:, 75:120], right_hand_pose=fp[:, 120:165],
            transl=tr, betas=torch.zeros(n, 10), expression=torch.zeros(n, 10),
        )
    return out.joints.detach().numpy()[:, :22, :]


def symmetry_error(joints):
    """Per-frame L-R asymmetry: mirror each left joint's position (relative to pelvis,
    negate X) and compare to its right partner. joints: (F,22,3) -> (F,) error."""
    pel = joints[:, J['pelvis']:J['pelvis'] + 1]
    rel = joints - pel
    err = np.zeros(joints.shape[0])
    pairs = [('left_ankle', 'right_ankle'), ('left_knee', 'right_knee'),
             ('left_wrist', 'right_wrist'), ('left_elbow', 'right_elbow'),
             ('left_foot', 'right_foot'), ('left_shoulder', 'right_shoulder')]
    for ln, rn in pairs:
        l = rel[:, J[ln]].copy(); l[:, 0] *= -1  # mirror left across X
        r = rel[:, J[rn]]
        err += np.linalg.norm(l - r, axis=1)
    return err


def motion_score(joints):
    n = joints.shape[0]
    lo, hi = int(n * 0.15), int(n * 0.85)
    if hi - lo < 10:
        return None, None, None
    vel = np.linalg.norm(np.diff(joints[lo:hi], axis=0), axis=2).sum(axis=1)
    win = 9
    smoothed = np.convolve(vel, np.ones(win) / win, mode='valid')
    return smoothed, lo, win


def mirror_aa(a):
    return np.array([a[0], -a[1], -a[2]])


def symmetrize(aa66):
    """aa66: 22x3 array (global_orient + 21 body joints). Mirror-average L/R pairs."""
    aa = aa66.copy()
    for li, ri in LR_PAIRS:
        l, r = aa[li], aa[ri]
        l_sym = (l + mirror_aa(r)) / 2.0
        aa[li] = l_sym
        aa[ri] = mirror_aa(l_sym)
    return aa


def find_files(pose_id, debug):
    # reuse the exact source-file basenames already matched by the 2D pipeline for this pose id
    src = debug[pose_id]['source_file']
    variant_glob = src.rsplit('_stageii', 1)[0].rsplit('-a_stageii', 1)[0]
    # simplest robust approach: search the whole data dir for files sharing the pose-name portion
    prefix = src.split('-a_')[0].split('-b_')[0]
    matches = glob.glob(os.path.join(DATA_DIR, '*', prefix + '*.pkl'))
    if not matches:
        matches = glob.glob(os.path.join(DATA_DIR, '*', src))
    return matches


debug = json.load(open(DEBUG_JSON))
result = {}
report = {}

for pose_id in ACCEPTED:
    files = find_files(pose_id, debug)
    if not files:
        print(f"MISSING files for {pose_id}, falling back to old axis-angle extraction")
        continue

    is_sym = pose_id not in ASYMMETRIC
    best = None  # (score, aa_frame_values)
    for f in files:
        with open(f, 'rb') as fh:
            d = pickle.load(fh, encoding='latin1')
        joints = smplx_joints(d['fullpose'], d['trans'])
        smoothed, lo, win = motion_score(joints)
        if smoothed is None:
            continue
        candidate_idx = np.arange(len(smoothed)) + lo + win // 2  # absolute frame indices
        m = smoothed  # motion score per candidate frame (already aligned to candidate_idx)
        if is_sym:
            sym = symmetry_error(joints)[candidate_idx]
            # normalize both terms to comparable scale, combine
            m_n = (m - m.min()) / (m.max() - m.min() + 1e-9)
            s_n = (sym - sym.min()) / (sym.max() - sym.min() + 1e-9)
            combined = 0.5 * m_n + 0.5 * s_n
        else:
            combined = m
        best_local = int(np.argmin(combined))
        frame = int(candidate_idx[best_local])
        score = float(combined[best_local])
        if best is None or score < best[0]:
            best = (score, f, d['fullpose'], frame, joints)

    score, fname, fullpose, frame, joints = best
    lo_w, hi_w = max(0, frame - 10), min(fullpose.shape[0], frame + 11)
    aa_avg = fullpose[lo_w:hi_w, :66].mean(axis=0).reshape(22, 3)
    if is_sym:
        aa_avg = symmetrize(aa_avg)
    result[pose_id] = [round(float(v), 5) for v in aa_avg.reshape(-1)]
    report[pose_id] = {'file': os.path.basename(fname), 'frame': frame, 'score': round(score, 5), 'symmetrized': is_sym}
    print(f"{pose_id:28s} <- {os.path.basename(fname)[:55]:55s} frame {frame:4d} score {score:.4f} sym={is_sym}")

with open(OUT, 'w') as f:
    json.dump(result, f, indent=1)
with open(OUT_APP, 'w') as f:
    json.dump(result, f, indent=1)
print(f"\nWrote {len(result)} poses to {OUT} and {OUT_APP}")
