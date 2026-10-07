// Splices MOYO-derived held-pose keyframes into index.html's POSES data, for the poses that
// passed visual QA. Only replaces the LAST keyframe (the held pose) in each pose's `kf: [...]`
// array, leaving any earlier "entry" keyframe(s) untouched so transition animations still work.
const fs = require('fs');

const APP_PATH = 'C:/SAVRO/Yoga App/index.html';
const MOCAP_PATH = 'C:/SAVRO/moyo_toolkit_inspect/retargeted_poses.json';

// Passed visual QA against the app's own renderer - see conversation for the full review.
const ACCEPTED = [
  'adho-mukha-svanasana', 'ananda-balasana', 'ardha-chandrasana', 'ardha-matsyendrasana',
  'ardha-pincha-mayurasana', 'bakasana', 'chaturanga-dandasana', 'dandasana', 'dhanurasana',
  'gomukhasana', 'halasana', 'malasana', 'marjaryasana-bitilasana', 'natarajasana', 'navasana',
  'parsvottanasana', 'phalakasana', 'purvottanasana', 'sarvangasana', 'setu-bandhasana',
  'supta-baddha-konasana', 'tolasana', 'urdhva-dhanurasana', 'ustrasana', 'utkatasana',
  'uttana-shishosana', 'vasisthasana', 'viparita-karani', 'virabhadrasana-3', 'vrksasana',
];

const mocap = JSON.parse(fs.readFileSync(MOCAP_PATH, 'utf8'));
let src = fs.readFileSync(APP_PATH, 'utf8');

function findMatchingBracket(str, openIdx, openCh, closeCh) {
  let depth = 0;
  for (let i = openIdx; i < str.length; i++) {
    if (str[i] === openCh) depth++;
    else if (str[i] === closeCh) { depth--; if (depth === 0) return i; }
  }
  throw new Error('no matching bracket from ' + openIdx);
}

// Split a bracket-balanced comma-separated list (top-level commas only).
function splitTopLevel(str) {
  const parts = [];
  let depth = 0, start = 0;
  for (let i = 0; i < str.length; i++) {
    const c = str[i];
    if ('([{'.includes(c)) depth++;
    else if (')]}'.includes(c)) depth--;
    else if (c === ',' && depth === 0) { parts.push(str.slice(start, i)); start = i + 1; }
  }
  parts.push(str.slice(start));
  return parts.map(s => s.trim()).filter(Boolean);
}

let applied = 0, missing = [];
for (const id of ACCEPTED) {
  const angles = mocap[id];
  if (!angles) { missing.push(id); continue; }

  const idMarker = `id: '${id}'`;
  const idIdx = src.indexOf(idMarker);
  if (idIdx === -1) { missing.push(id + ' (id not found)'); continue; }

  // Find the P({ ... }) call this id belongs to, by walking back to the nearest 'P({' and
  // forward to its matching close, using paren balance starting at that '('.
  const pOpenIdx = src.lastIndexOf('P({', idIdx);
  const parenClose = findMatchingBracket(src, pOpenIdx + 1, '(', ')');

  const block = src.slice(pOpenIdx, parenClose + 1);
  const kfIdx = block.indexOf('kf:');
  const arrOpen = block.indexOf('[', kfIdx);
  const arrClose = findMatchingBracket(block, arrOpen, '[', ']');
  const arrInner = block.slice(arrOpen + 1, arrClose);

  const frames = splitTopLevel(arrInner);
  const newLast = '[' + angles.map(a => a.toFixed(1)).join(', ') + ']';
  frames[frames.length - 1] = newLast;
  const newArrInner = frames.join(', ');

  const newBlock = block.slice(0, arrOpen + 1) + newArrInner + block.slice(arrClose);
  src = src.slice(0, pOpenIdx) + newBlock + src.slice(parenClose + 1);
  applied++;
  console.log(`applied ${id}: ${frames.length} keyframe(s), held = ${newLast}`);
}

fs.writeFileSync(APP_PATH, src);
console.log(`\nApplied ${applied}/${ACCEPTED.length}.`);
if (missing.length) console.log('MISSING:', missing.join(', '));
