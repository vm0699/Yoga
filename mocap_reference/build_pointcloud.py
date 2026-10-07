"""
Point-cloud avatar generator.

Samples ~100k points on the SMPL-X body surface (shared by every asana) and bakes a real MoYo mocap
sequence (entry -> hold -> exit) into a compact pose track the app can play back with GPU skinning.

Outputs (all into <app>/assets3d/):
  cloud_body.bin / cloud_body.json   shared dots: rest position(int16x3), normal(int8x3),
                                     skin joints(uint8x4), skin weights(uint8x4)  [22-joint rig]
  cloud_<pose>.bin / cloud_<pose>.json   per-asana track: float32 frames x 66 axis-angle (30 fps),
                                         per-frame ground offset, hold window, framing bounds
Optional: exports/<pose>_hold.ply - the held frame as a coloured point cloud (opens in AutoCAD/ReCap/MeshLab).

Usage: python build_pointcloud.py <app_pose_id> <mocap_file_glob> [N_POINTS]
"""
import glob, json, os, pickle, struct, sys
import numpy as np

MODEL = r"C:\SAVRO\smplx_models\standard\models\smplx\SMPLX_NEUTRAL.npz"
DATA_DIR = r"C:\SAVRO\moyo_toolkit_inspect\data\mosh_extracted\mosh"
APP = r"C:\SAVRO\Yoga App"
OUT = os.path.join(APP, "assets3d")
EXPORTS = os.path.join(APP, "exports")
os.makedirs(OUT, exist_ok=True); os.makedirs(EXPORTS, exist_ok=True)

pose_id = sys.argv[1] if len(sys.argv) > 1 else "vrksasana"
file_glob = sys.argv[2] if len(sys.argv) > 2 else "*Tree_Pose*-a_stageii.pkl"
N = int(sys.argv[3]) if len(sys.argv) > 3 else 100000
OUT_FPS = 30

d = np.load(MODEL, allow_pickle=True)
V = d['v_template'].astype(np.float64)
F = d['f'].astype(np.int64)
W55 = d['weights'].astype(np.float64)
Jreg = d['J_regressor'].astype(np.float64)
parents = d['kintree_table'][0].astype(np.int64)
parents = np.where(parents > 1000, -1, parents)  # root sentinel
J55 = Jreg @ V

# ---- remap the 55-joint skin weights onto the 22 body joints (hands/jaw/eyes are never posed, so they
# ---- move rigidly with their parent: jaw/eyes -> head(15), left hand -> wrist 20, right hand -> wrist 21)
REMAP = list(range(22)) + [15, 15, 15] + [20] * 15 + [21] * 15
W22 = np.zeros((V.shape[0], 22))
for j in range(55):
    W22[:, REMAP[j]] += W55[:, j]

# ---- vertex normals
tri = V[F]
fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
vn = np.zeros_like(V)
for i in range(3):
    np.add.at(vn, F[:, i], fn)
vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)

# ---- area-weighted surface sampling
rng = np.random.default_rng(7)
area = 0.5 * np.linalg.norm(fn, axis=1)
cum = np.cumsum(area); cum /= cum[-1]
fi = np.searchsorted(cum, rng.random(N))
r1, r2 = rng.random(N), rng.random(N)
s = np.sqrt(r1)
b0, b1, b2 = 1 - s, s * (1 - r2), s * r2
fv = F[fi]
bary = np.stack([b0, b1, b2], 1)
P = (V[fv] * bary[:, :, None]).sum(1)
Nrm = (vn[fv] * bary[:, :, None]).sum(1)
Nrm /= np.maximum(np.linalg.norm(Nrm, axis=1, keepdims=True), 1e-12)
Wp = (W22[fv] * bary[:, :, None]).sum(1)                      # (N,22)
top = np.argsort(-Wp, axis=1)[:, :4]
tw = np.take_along_axis(Wp, top, axis=1)
tw /= np.maximum(tw.sum(1, keepdims=True), 1e-12)

SCALE = 2.0  # positions stored as int16 over [-2, 2] m
pos_q = np.clip(np.round(P / SCALE * 32767), -32767, 32767).astype('<i2')
nrm_q = np.clip(np.round(Nrm * 127), -127, 127).astype('i1')
jnt_q = top.astype('u1')
wgt_q = np.clip(np.round(tw * 255), 0, 255).astype('u1')

body_bin = os.path.join(OUT, "cloud_body.bin")
with open(body_bin, "wb") as f:
    f.write(pos_q.tobytes()); f.write(nrm_q.tobytes()); f.write(jnt_q.tobytes()); f.write(wgt_q.tobytes())
json.dump({"n": int(N), "posScale": SCALE, "bytes": os.path.getsize(body_bin)}, open(os.path.join(OUT, "cloud_body.json"), "w"))
print(f"cloud_body.bin: {N} dots, {os.path.getsize(body_bin)/1e6:.2f} MB")


# ---- forward kinematics identical to the app's skinning (22 joints, rest-pose bind, no blendshapes)
def rodrigues(aa):
    th = np.linalg.norm(aa, axis=-1, keepdims=True)
    k = aa / np.maximum(th, 1e-9)
    K = np.zeros(aa.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2] = -k[..., 2], k[..., 1]
    K[..., 1, 0], K[..., 1, 2] = k[..., 2], -k[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -k[..., 1], k[..., 0]
    th = th[..., None]
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def fk(aa22):  # (F,22,3) -> A (F,22,4,4) global joint transforms
    Fn = aa22.shape[0]
    R = rodrigues(aa22)
    A = np.zeros((Fn, 22, 4, 4)); A[..., 3, 3] = 1
    for j in range(22):
        loc = np.zeros((Fn, 4, 4)); loc[:, :3, :3] = R[:, j]; loc[:, 3, 3] = 1
        loc[:, :3, 3] = J55[j] - (J55[parents[j]] if parents[j] >= 0 else 0)
        A[:, j] = loc if parents[j] < 0 else A[:, parents[j]] @ loc
    return A


# ---- load + resample the real mocap sequence
fp = glob.glob(os.path.join(DATA_DIR, "*", file_glob))[0]
m = pickle.load(open(fp, "rb"), encoding="latin1")
fps_in = float(m['stageii_debug_details']['mocap_frame_rate'])
step = max(1, int(round(fps_in / OUT_FPS)))
aa = m['fullpose'][::step, :66].astype(np.float64).reshape(-1, 22, 3)
nF = aa.shape[0]
print(f"{os.path.basename(fp)}: {m['fullpose'].shape[0]} frames @ {fps_in:g} fps -> {nF} frames @ {fps_in/step:g} fps")

A = fk(aa)
jw = A[:, :, :3, 3]                                          # joint world positions (Z-up capture frame)

# ---- trim to the actual action: drop the idle lead-in / lead-out
dev_start = np.linalg.norm(jw - jw[0:1], axis=2).mean(1)
dev_end = np.linalg.norm(jw - jw[-1:], axis=2).mean(1)
thr = 0.04
s0 = max(0, int(np.argmax(dev_start > thr)) - 8)
e0 = min(nF - 1, nF - 1 - int(np.argmax(dev_end[::-1] > thr)) + 8)
aa, A, jw = aa[s0:e0 + 1], A[s0:e0 + 1], jw[s0:e0 + 1]
nF = aa.shape[0]
print(f"trimmed to frames {s0}..{e0} -> {nF} frames ({nF/OUT_FPS:.1f}s)")

# ---- hold window: calmest stretch after the pose has been reached
vel = np.r_[0, np.linalg.norm(np.diff(jw, axis=0), axis=2).sum(1)]
sm = np.convolve(vel, np.ones(9) / 9, mode='same')
reach = np.linalg.norm(jw - jw[0:1], axis=2).mean(1)
cand = np.where(reach > 0.7 * reach.max())[0]
hold_f = int(cand[np.argmin(sm[cand])])
lim = max(sm[hold_f] * 2.5, 0.02)
hs = hold_f
while hs > 0 and sm[hs - 1] < lim: hs -= 1
he = hold_f
while he < nF - 1 and sm[he + 1] < lim: he += 1
print(f"hold frame {hold_f}, window {hs}..{he} ({(he-hs)/OUT_FPS:.1f}s)")

# ---- facing: upright poses are shown from the front, lying/folded poses from the side
lr = jw[hold_f, 1, :2] - jw[hold_f, 2, :2]; lr /= max(np.linalg.norm(lr), 1e-9)       # left hip - right hip (capture frame)
spine = jw[hold_f, 9] - jw[hold_f, 0]; upright = abs(spine[2]) / max(np.linalg.norm(spine), 1e-9)
vx, vz = (lr[1], lr[0]) if upright > 0.6 else (lr[0], -lr[1])                          # world (x,z) of the direction to face the camera
yaw = float(np.arctan2(-vx, vz)); cy, sy = np.cos(yaw), np.sin(yaw)
print(f"upright {upright:.2f} -> {'front' if upright > 0.6 else 'side'} view, yaw {np.degrees(yaw):.0f} deg")

# ---- grounding + framing: skin the real mesh (22-joint LBS) per frame, lowest point = floor
M = A.copy()
M[:, :, :3, 3] -= np.einsum('fjab,jb->fja', A[:, :, :3, :3], J55[:22])  # A_j * T(-J_j)
ground = np.zeros(nF); mn = np.full(3, 1e9); mx = np.full(3, -1e9)
for f0 in range(nF):
    Rm, tm = M[f0, :, :3, :3], M[f0, :, :3, 3]
    out = np.zeros_like(V)
    for j in range(22):
        out += W22[:, j:j + 1] * (V @ Rm[j].T + tm[j])
    ground[f0] = out[:, 2].min()                             # Z-up: lowest vertex
    wx, wz, wy = out[:, 0], -out[:, 1], out[:, 2] - ground[f0]
    fx_, fz_ = wx * cy + wz * sy, -wx * sy + wz * cy
    mn = np.minimum(mn, [fx_.min(), wy.min(), fz_.min()]); mx = np.maximum(mx, [fx_.max(), wy.max(), fz_.max()])
    if f0 == hold_f:
        hold_verts = out.copy()
bounds = {"minX": float(mn[0]), "maxX": float(mx[0]), "minY": float(mn[1]), "maxY": float(mx[1]),
          "minZ": float(mn[2]), "maxZ": float(mx[2])}

pose_bin = os.path.join(OUT, f"cloud_{pose_id}.bin")
aa.astype('<f4').tofile(pose_bin)
meta = {"id": pose_id, "frames": int(nF), "fps": OUT_FPS, "hold": [int(hs), int(hold_f), int(he)], "yaw": round(yaw, 4),
        "ground": [round(float(g), 4) for g in ground], "bounds": bounds, "source": os.path.basename(fp)}
json.dump(meta, open(os.path.join(OUT, f"cloud_{pose_id}.json"), "w"))
print(f"cloud_{pose_id}.bin: {os.path.getsize(pose_bin)/1e3:.0f} KB; bounds {bounds}")

# ---- optional PLY of the held pose (Y-up, metres) for AutoCAD / ReCap / MeshLab
Rm, tm = M[hold_f, :, :3, :3], M[hold_f, :, :3, 3]
sk = np.zeros((N, 3))
for k in range(4):
    j = top[:, k]
    sk += tw[:, k:k + 1] * (np.einsum('nab,nb->na', Rm[j], P) + tm[j])
sk[:, 2] -= ground[hold_f]
yup = np.stack([sk[:, 0], sk[:, 2], -sk[:, 1]], 1)
t = (yup[:, 1] - yup[:, 1].min()) / max(np.ptp(yup[:, 1]), 1e-6)
col = np.stack([60 + 150 * t, 170 - 30 * t, 220 - 120 * t], 1).astype(int)
ply = os.path.join(EXPORTS, f"{pose_id}_hold.ply")
with open(ply, "w") as f:
    f.write(f"ply\nformat ascii 1.0\nelement vertex {N}\nproperty float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    for p, c in zip(yup, col):
        f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {c[0]} {c[1]} {c[2]}\n")
print("PLY:", ply)

mf = os.path.join(OUT, "cloud_manifest.json")
ids = json.load(open(mf))["ids"] if os.path.exists(mf) else []
if pose_id not in ids: ids.append(pose_id)
json.dump({"ids": sorted(ids)}, open(mf, "w"))
print("manifest:", sorted(ids))
