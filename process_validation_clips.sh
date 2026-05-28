#!/bin/bash
# Process the 10 wide-angle labeled clips used to validate formation recognition.
# Runs the full YOLO pipeline (via run_one_clip.sh) on each clip into
# cache/CSAI_FORMATIONS/. REQUIRES torch + ultralytics in your active python3
# env (the same env the desktop app uses) -- the analysis sandbox has neither.
#
#   ./process_validation_clips.sh
#
# Then run the comparison (torch-free):
#   python formations/validate_formations.py --folder CSAI_FORMATIONS
set -e
cd "$(dirname "$0")"

CLIPS=(
  "Wide - Clip 275"   # DETROIT
  "Wide - Clip 188"   # TRIPS OPEN
  "Wide - Clip 854"   # TREY Y OFF
  "Wide - Clip 114"   # TRIPS
  "Wide - Clip 933"   # DALLAS Y OFF
  "Wide - Clip 278"   # DALLAS
  "Wide - Clip 305"   # TREY
  "Wide - Clip 564"   # DENVER
  "Wide - Clip 1021"  # DALLAS WG
  "Wide - Clip 021"   # TREY Y OFF TITE
)

for clip in "${CLIPS[@]}"; do
  video="data/CSAI_FORMATIONS/${clip}.mp4"
  if [[ ! -f "$video" ]]; then
    echo "!! missing video: $video -- skipping"
    continue
  fi
  echo "########## Processing: $clip ##########"
  # Don't let one clip's failure abort the whole batch (set -e) — log and continue
  # so the comparison still runs on whatever processed.
  ./run_one_clip.sh "$video" || echo "!! failed to process: $clip (continuing)"
done

echo
echo "########## All clips processed — running comparison ##########"
# The validator is torch-free; it reads the cache JSON the steps above wrote.
python3 formations/validate_formations.py --folder CSAI_FORMATIONS

echo
echo "########## Backfilling data CSV so the app overlay shows the new predictions ##########"
python3 - <<'PY'
import sys, os, glob, pandas as pd
sys.path.insert(0, 'formations')
import template_matcher as tm
folder='CSAI_FORMATIONS'; base='cache'
data_csv=f'{base}/{folder}/{folder}_data.csv'
df=pd.read_csv(data_csv)
for c in ('TEMPLATE FORM','TEMPLATE SCORE','TEMPLATE RELIABLE','TEMPLATE METHOD'):
    if c not in df.columns: df[c]=''
processed=sorted({os.path.basename(p).replace('_position.json','')
                  for p in glob.glob(f'{base}/{folder}/positions/*_position.json')})
for clip in processed:
    res=tm.recognize_from_cache(clip, folder, base)
    if not res.get('formation'): continue
    idx=df.index[df['CLIP NAME'].astype(str)==clip].tolist()
    if not idx:
        df=pd.concat([df, pd.DataFrame([{'CLIP NAME':clip}])], ignore_index=True); idx=[df.index[-1]]
    i=idx[0]
    df.at[i,'TEMPLATE FORM']=res['formation']
    df.at[i,'TEMPLATE SCORE']=res['score']
    df.at[i,'TEMPLATE RELIABLE']=bool(res['reliable'])
    df.at[i,'TEMPLATE METHOD']=res.get('method','legacy')
df.to_csv(data_csv, index=False)
print(f'Backfilled {len(processed)} clips into {data_csv}')
PY
