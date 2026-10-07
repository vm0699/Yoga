# MOYO motion-capture pose reference

Small (few hundred KB) intermediate output from the MOYO dataset retargeting pass done 2026-09-28.
Nothing here is the MOYO/SMPL-X dataset itself - just derived joint-angle numbers, safe to keep.

- `retargeted_poses.json` - the 12-angle keyframe extracted for all 45 name-matched poses
  (30 of which are actually applied in index.html; see ACCEPTED list in apply_to_app.js).
- `retargeted_debug.json` - which source file/frame each pose came from, for traceability.
- `retarget.py` - the extraction script (needs the full MOYO mosh.zip + SMPL-X model files
  to rerun - those are NOT kept locally, see below).
- `apply_to_app.js` - splices retargeted_poses.json into index.html's POSES data.

To add more poses later or redo one: re-download mosh.zip + SMPL-X models (see prior chat),
rerun retarget.py, then apply_to_app.js with an updated ACCEPTED list.
