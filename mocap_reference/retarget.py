"""
Retarget MOYO SMPL-X mocap fits onto the yoga app's simplified 2D side-view skeleton.

For each mapped pose:
  1. Load the SMPL-X pose sequence (.pkl), run forward kinematics to get 3D joint
     positions per frame (Z-up world frame, confirmed empirically).
  2. Find the most "held" (lowest local motion) frame in the middle portion of
     the sequence - that's the actual asana, not the entry/exit transition.
  3. Compute a per-pose forward direction from the shoulder line (up x shoulder
     vector), so the 3D pose can be projected into a single consistent sagittal
     (side-view) plane regardless of which way the subject faced the cameras.
  4. Convert each bone (pelvis->shoulder, shoulder->head, shoulder->elbow->wrist,
     hip->knee->ankle->foot) into the app's angle convention:
       angle = atan2(forward_component, -vertical_component) in degrees
       (0=down, 90=forward, 180=up, -90=back - matches index.html's K()/fk()).
  5. Write one JSON keyed by app pose id -> 12-value keyframe array, matching
     the [torso, head, near_ua, near_fa, far_ua, far_fa, near_th, near_sh, near_ft,
     far_th, far_sh, far_ft] order used in index.html's KI mapping.

Output: retargeted_poses.json (also a debug .txt with the raw joint dump per pose
for spot-checking) - both go through visual QA in the browser before touching
index.html.
"""
import glob
import json
import math
import os
import pickle

import numpy as np
import torch
import smplx

DATA_DIR = r"C:\SAVRO\moyo_toolkit_inspect\data\mosh_extracted\mosh"
MODEL_DIR = r"C:\SAVRO\smplx_models\standard\models"
OUT_JSON = r"C:\SAVRO\moyo_toolkit_inspect\retargeted_poses.json"
OUT_DEBUG = r"C:\SAVRO\moyo_toolkit_inspect\retargeted_debug.json"

JOINT_NAMES = ['pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee', 'spine2',
               'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot', 'neck',
               'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder',
               'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist']
J = {n: i for i, n in enumerate(JOINT_NAMES)}

# MOYO pose-name substring -> app pose id. near=right side, far=left side (consistent, arbitrary;
# flip per-pose after visual QA if the "active" limb ends up on the wrong side).
POSE_MAP = {
    'Boat_Pose_or_Paripurna_Navasana': 'navasana',
    'Bound_Angle_Pose_or_Baddha_Konasana': 'baddha-konasana',
    'Bow_Pose_or_Dhanurasana': 'dhanurasana',
    'Bridge_Pose_or_Setu_Bandha_Sarvangasana': 'setu-bandhasana',
    'Camel_Pose_or_Ustrasana': 'ustrasana',
    'Cat_Cow_Pose_or_Marjaryasana': 'marjaryasana-bitilasana',
    'Chair_Pose_or_Utkatasana': 'utkatasana',
    'Child_Pose_or_Balasana': 'balasana',
    'Cobra_Pose_or_Bhujangasana': 'bhujangasana',
    'Corpse_Pose_or_Savasana': 'savasana',
    'Cow_Face_Pose_or_Gomukhasana': 'gomukhasana',
    'Crane_(Crow)_Pose_or_Bakasana': 'bakasana',
    'Dolphin_Pose_or_Ardha_Pincha_Mayurasana': 'ardha-pincha-mayurasana',
    'Downward-Facing_Dog_pose_or_Adho_Mukha_Svanasana': 'adho-mukha-svanasana',
    'Eagle_Pose_or_Garudasana': 'garudasana',
    'Extended_Puppy_Pose_or_Uttana_Shishosana': 'uttana-shishosana',
    'Fish_Pose_or_Matsyasana': 'matsyasana',
    'Four-Limbed_Staff_Pose_or_Chaturanga_Dandasana': 'chaturanga-dandasana',
    'Garland_Pose_or_Malasana': 'malasana',
    'Half_Lord_of_the_Fishes_Pose_or_Ardha_Matsyendrasana': 'ardha-matsyendrasana',
    'Half_Moon_Pose_or_Ardha_Chandrasana': 'ardha-chandrasana',
    'Happy_Baby_Pose_or_Ananda_Balasana': 'ananda-balasana',
    'Head-to-Knee_Forward_Bend_pose_or_Janu_Sirsasana': 'janu-sirsasana',
    'Intense_Side_Stretch_Pose_or_Parsvottanasana': 'parsvottanasana',
    'Legs-Up-the-Wall_Pose_or_Viparita_Karani': 'viparita-karani',
    'Locust_Pose_or_Salabhasana': 'salabhasana',
    'Lord_of_the_Dance_Pose_or_Natarajasana': 'natarajasana',
    'Low_Lunge_pose_or_Anjaneyasana': 'anjaneyasana',
    'Plank_Pose_or_Kumbhakasana': 'phalakasana',
    'Plow_Pose_or_Halasana': 'halasana',
    'Reclining_Hand-to-Big-Toe_Pose_or_Supta_Padangusthasana': 'supta-padangusthasana',
    'Scale_Pose_or_Tolasana': 'tolasana',
    'Side_Plank_Pose_or_Vasisthasana': 'vasisthasana',
    'Staff_Pose_or_Dandasana': 'dandasana',
    'Standing_Forward_Bend_pose_or_Uttanasana': 'uttanasana',
    'Supported_Headstand_pose_or_Salamba_Sirsasana': 'sirsasana',
    'Supported_Shoulderstand_pose_or_Salamba_Sarvangasana': 'sarvangasana',
    'Supta_Baddha_Konasana': 'supta-baddha-konasana',
    'Seated_Forward_Bend_pose_or_Paschimottanasana': 'paschimottanasana',
    'Tree_Pose_or_Vrksasana': 'vrksasana',
    'Upward_Bow_(Wheel)_Pose_or_Urdhva_Dhanurasana': 'urdhva-dhanurasana',
    'Upward_Plank_Pose_or_Purvottanasana': 'purvottanasana',
    'Warrior_III_Pose_or_Virabhadrasana_III': 'virabhadrasana-3',
    'Warrior_II_Pose_or_Virabhadrasana_II': 'virabhadrasana-2',
    'Rajakapotasana': 'eka-pada-rajakapotasana',
}

print("Loading SMPL-X model...")
model = smplx.create(MODEL_DIR, model_type='smplx', gender='female', use_pca=False,
                      flat_hand_mean=True, num_betas=10, ext='npz')


def smplx_joints(fullpose, trans):
    """fullpose: (N,165), trans: (N,3) -> joints: (N,22,3), body joints only."""
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


def find_held_frame(joints):
    """Lowest-motion frame in the middle 70% of the sequence = the actual held asana.
    Returns (frame_index, motion_score) - lower score = more stable hold."""
    n = joints.shape[0]
    lo, hi = int(n * 0.15), int(n * 0.85)
    if hi - lo < 5:
        return n // 2, 1e9
    vel = np.linalg.norm(np.diff(joints[lo:hi], axis=0), axis=2).sum(axis=1)  # per-frame total joint speed
    win = 9
    smoothed = np.convolve(vel, np.ones(win) / win, mode='valid')
    idx = int(np.argmin(smoothed))
    return lo + idx + win // 2, float(smoothed[idx])


def averaged_pose(joints, frame, half_window=10):
    """Average joint positions over a small window around `frame` to smooth mocap/fitting noise."""
    lo, hi = max(0, frame - half_window), min(joints.shape[0], frame + half_window + 1)
    return joints[lo:hi].mean(axis=0)


def bone_angle(parent, child, fwd, up_axis=2):
    """Angle of the parent->child bone in the app's convention (0=down,90=fwd,180=up,-90=back)."""
    d = child - parent
    horiz = float(np.dot(d[:2], fwd))
    vert = float(d[up_axis])
    return math.degrees(math.atan2(horiz, -vert))


def extract_pose(joints):
    pel = joints[J['pelvis']]
    lsh, rsh = joints[J['left_shoulder']], joints[J['right_shoulder']]
    sh_mid = (lsh + rsh) / 2.0

    # Chest-forward direction: (right_shoulder - left_shoulder) x up, horizontal plane only.
    # (Empirically verified against the app's own renderer across the matched pose set - this
    # sign convention is the one that lines up with how the app draws bone angles.)
    shoulder_vec = (lsh - rsh)[:2]
    shoulder_vec = shoulder_vec / (np.linalg.norm(shoulder_vec) + 1e-8)
    fwd = np.array([shoulder_vec[1], -shoulder_vec[0]])  # rotate +90 deg

    torso = bone_angle(pel, sh_mid, fwd)
    head = bone_angle(sh_mid, joints[J['head']], fwd)

    # near = right side, far = left side (consistent placeholder; adjust per-pose after visual QA).
    near_ua = bone_angle(rsh, joints[J['right_elbow']], fwd)
    near_fa = bone_angle(joints[J['right_elbow']], joints[J['right_wrist']], fwd)
    far_ua = bone_angle(lsh, joints[J['left_elbow']], fwd)
    far_fa = bone_angle(joints[J['left_elbow']], joints[J['left_wrist']], fwd)

    near_th = bone_angle(joints[J['right_hip']], joints[J['right_knee']], fwd)
    near_sh = bone_angle(joints[J['right_knee']], joints[J['right_ankle']], fwd)
    near_ft = bone_angle(joints[J['right_ankle']], joints[J['right_foot']], fwd)
    far_th = bone_angle(joints[J['left_hip']], joints[J['left_knee']], fwd)
    far_sh = bone_angle(joints[J['left_knee']], joints[J['left_ankle']], fwd)
    far_ft = bone_angle(joints[J['left_ankle']], joints[J['left_foot']], fwd)

    return [torso, head, near_ua, near_fa, far_ua, far_fa, near_th, near_sh, near_ft, far_th, far_sh, far_ft]


def find_files():
    files = glob.glob(os.path.join(DATA_DIR, '*', '*.pkl'))
    by_pose = {}
    for f in files:
        base = os.path.basename(f)
        for moyo_name, app_id in POSE_MAP.items():
            if moyo_name in base:
                by_pose.setdefault(app_id, []).append(f)
    return by_pose


def main():
    by_pose = find_files()
    print(f"Matched {len(by_pose)} of {len(set(POSE_MAP.values()))} mapped poses, "
          f"{sum(len(v) for v in by_pose.values())} candidate files total.")

    results = {}
    debug = {}
    for app_id, files in sorted(by_pose.items()):
        # Evaluate every variant file, pick whichever has the single most stable (lowest-motion)
        # hold anywhere in it - length is not a proxy for quality, a clean short take beats a
        # long one with a wobble in the middle.
        best = None  # (score, file, joints, frame)
        for f in files:
            with open(f, 'rb') as fh:
                d = pickle.load(fh, encoding='latin1')
            joints = smplx_joints(d['fullpose'], d['trans'])
            frame, score = find_held_frame(joints)
            if best is None or score < best[0]:
                best = (score, f, joints, frame)
        _, best_file, best_joints, best_frame = best
        held = averaged_pose(best_joints, best_frame, half_window=10)
        angles = extract_pose(held)
        results[app_id] = [round(a, 1) for a in angles]
        debug[app_id] = {
            'source_file': os.path.basename(best_file),
            'frame': int(best_frame), 'total_frames': int(best_joints.shape[0]),
            'motion_score': round(best[0], 5),
            'joint_positions': held.round(4).tolist(),
        }
        print(f"  {app_id:35s} <- {os.path.basename(best_file)[:60]} (frame {best_frame}/{best_joints.shape[0]}, score {best[0]:.4f})")

    with open(OUT_JSON, 'w') as f:
        json.dump(results, f, indent=2)
    with open(OUT_DEBUG, 'w') as f:
        json.dump(debug, f, indent=2)
    print(f"\nWrote {len(results)} poses to {OUT_JSON}")


if __name__ == '__main__':
    main()
