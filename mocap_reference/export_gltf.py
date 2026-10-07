"""
Export the neutral SMPL-X body (rest pose, no shape/pose blendshapes) as a rigged,
skinned GLB for use in Three.js. Only the 22 body joints are meaningfully driven by
the app; hand/jaw/eye joints are kept in the skeleton (for correct skin deformation)
but always at rest rotation.

Output: C:/SAVRO/Yoga App/assets3d/body.glb
"""
import struct
import numpy as np
from pygltflib import (
    GLTF2, Asset, Scene, Node, Mesh, Primitive, Attributes, Buffer, BufferView,
    Accessor, Skin, Material, PbrMetallicRoughness, ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER,
    FLOAT, UNSIGNED_SHORT, UNSIGNED_INT, VEC3, VEC4, SCALAR, MAT4
)

MODEL = r"C:\SAVRO\smplx_models\standard\models\smplx\SMPLX_NEUTRAL.npz"
OUT = r"C:\SAVRO\Yoga App\assets3d\body.glb"
N_JOINTS = 55  # full skeleton kept for correct skinning; only 0-21 are ever rotated

d = np.load(MODEL, allow_pickle=True)
v_template = d['v_template'].astype(np.float32)          # (10475,3)
faces = d['f'].astype(np.uint32)                          # (20908,3)
weights_full = d['weights'].astype(np.float32)            # (10475,55)
kintree = d['kintree_table']                               # (2,55)
J_regressor = d['J_regressor'].astype(np.float32)          # (55,10475)
J = (J_regressor @ v_template).astype(np.float32)          # (55,3) rest joint positions

parents = kintree[0].astype(np.int64)
parents = np.where(parents == parents.max(), -1, parents)  # root sentinel -> -1

n_verts = v_template.shape[0]

# --- vertex normals (smooth, area-weighted face-normal accumulation) ---
normals = np.zeros_like(v_template)
tris = v_template[faces]
fn = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
for i in range(3):
    np.add.at(normals, faces[:, i], fn)
lens = np.linalg.norm(normals, axis=1, keepdims=True)
lens[lens == 0] = 1
normals = (normals / lens).astype(np.float32)

# --- top-4 skinning influences per vertex (glTF limit), renormalized ---
top4_idx = np.argsort(-weights_full, axis=1)[:, :4].astype(np.uint16)  # (N,4)
top4_w = np.take_along_axis(weights_full, top4_idx, axis=1)
row_sum = top4_w.sum(axis=1, keepdims=True)
row_sum[row_sum == 0] = 1
top4_w = (top4_w / row_sum).astype(np.float32)

# --- inverse bind matrices: rest pose has no rotation, so this is just -J[i] translation ---
ibm = np.zeros((N_JOINTS, 4, 4), dtype=np.float32)
for i in range(N_JOINTS):
    m = np.eye(4, dtype=np.float32)
    m[:3, 3] = -J[i]
    ibm[i] = m

# --- pack all binary buffers, 4-byte aligned ---
def pad4(buf):
    while len(buf) % 4 != 0:
        buf += b'\x00'
    return buf

blob = b''
views = []  # (offset, length, target)

def add(data, target=None):
    global blob
    offset = len(blob)
    blob = pad4(blob + data.tobytes())
    views.append((offset, data.nbytes, target))
    return len(views) - 1

vi_pos = add(v_template, ARRAY_BUFFER)
vi_nrm = add(normals, ARRAY_BUFFER)
vi_joints = add(top4_idx, ARRAY_BUFFER)
vi_weights = add(top4_w, ARRAY_BUFFER)
vi_idx = add(faces.reshape(-1), ELEMENT_ARRAY_BUFFER)
vi_ibm = add(np.ascontiguousarray(ibm.transpose(0, 2, 1)).reshape(-1, 16))  # glTF matrices are column-major

gltf = GLTF2()
gltf.asset = Asset(version="2.0", generator="yoga-app custom SMPL-X exporter")
gltf.buffers = [Buffer(byteLength=len(blob))]
gltf.bufferViews = [
    BufferView(buffer=0, byteOffset=o, byteLength=l, target=t) if t else BufferView(buffer=0, byteOffset=o, byteLength=l)
    for (o, l, t) in views
]

def accessor(view_idx, count, type_, comp_type, mins=None, maxs=None, normalized=False):
    return Accessor(bufferView=view_idx, componentType=comp_type, count=count, type=type_,
                     min=mins, max=maxs, normalized=normalized)

gltf.accessors = [
    accessor(vi_pos, n_verts, VEC3, FLOAT, v_template.min(axis=0).tolist(), v_template.max(axis=0).tolist()),  # 0 POSITION
    accessor(vi_nrm, n_verts, VEC3, FLOAT),                                                                     # 1 NORMAL
    accessor(vi_joints, n_verts, VEC4, 5123),                                                                   # 2 JOINTS_0 (unsigned short)
    accessor(vi_weights, n_verts, VEC4, FLOAT),                                                                 # 3 WEIGHTS_0
    accessor(vi_idx, faces.size, SCALAR, UNSIGNED_INT),                                                         # 4 indices
    accessor(vi_ibm, N_JOINTS, MAT4, FLOAT),                                                                    # 5 inverseBindMatrices
]

gltf.materials = [Material(
    pbrMetallicRoughness=PbrMetallicRoughness(baseColorFactor=[0.20, 0.47, 0.70, 1.0], metallicFactor=0.05, roughnessFactor=0.55),
    doubleSided=False,
)]

gltf.meshes = [Mesh(primitives=[Primitive(
    attributes=Attributes(POSITION=0, NORMAL=1, JOINTS_0=2, WEIGHTS_0=3),
    indices=4, material=0,
)])]

# joint nodes: node i's local translation = J[i] - J[parent[i]] (or J[i] for the root)
joint_node_start = 1  # node 0 will be the mesh node
nodes = [Node(mesh=0, skin=0)]  # node 0: the skinned mesh
children_of = {i: [] for i in range(N_JOINTS)}
for i, p in enumerate(parents):
    if p >= 0:
        children_of[int(p)].append(i)

for i in range(N_JOINTS):
    local_t = (J[i] - J[int(parents[i])]) if parents[i] >= 0 else J[i]
    nodes.append(Node(name=f"joint_{i}", translation=local_t.tolist(),
                       children=[joint_node_start + c for c in children_of[i]] or None))

gltf.nodes = nodes
gltf.skins = [Skin(inverseBindMatrices=5, joints=[joint_node_start + i for i in range(N_JOINTS)], skeleton=joint_node_start + 0)]
gltf.scenes = [Scene(nodes=[0, joint_node_start + 0])]  # mesh node + root joint node, both at scene root
gltf.scene = 0

gltf.set_binary_blob(blob)
gltf.save_binary(OUT)
print("Wrote", OUT, "-", n_verts, "verts,", faces.shape[0], "tris,", N_JOINTS, "joints")

import os
print("File size:", round(os.path.getsize(OUT) / 1024 / 1024, 2), "MB")
