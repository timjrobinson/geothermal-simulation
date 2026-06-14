#!/usr/bin/env bash
# Reproducible downloader for the Utah FORGE real test dataset (DOE GDR, CC-BY 4.0).
# All files are co-located over the FORGE footprint. Usage:
#   ./fetch.sh            # everything, incl. the multi-GB well-log archives (~13 GB)
#   ./fetch.sh --small    # skip the large well-log/InSAR archives (~250 MB total)
set -uo pipefail
cd "$(dirname "$0")"
M=measured
mkdir -p "$M"/{gravity,em,mt/edi,welllog/16A,welllog/58-32,insar,microseismic,seismic}
SMALL=0; [ "${1:-}" = "--small" ] && SMALL=1

get(){ # dest, url
  [ -s "$1" ] && { echo "  have $1"; return; }
  echo "  GET $(basename "$1")"
  curl -fLsS -C - -o "$1" "$2" && echo "    OK $(du -h "$1" | cut -f1)" || echo "    FAIL $1"
}

echo "== gravity + TEM (GDR 1002) =="
get "$M/gravity/Utah_FORGE_Gravity_Data.zip" "https://gdr.openei.org/files/1002/Utah_FORGE_Gravity_Data%20(1).zip"
get "$M/em/Utah_FORGE_TEM_USF.zip"           "https://gdr.openei.org/files/1002/Utah_FORGE_TEM_USF.zip"
( cd "$M/gravity" && unzip -o -q Utah_FORGE_Gravity_Data.zip && rm -f Utah_FORGE_Gravity_Data.zip )
( cd "$M/em"      && unzip -o -q Utah_FORGE_TEM_USF.zip && rm -f Utah_FORGE_TEM_USF.zip )

echo "== magnetotellurics (GDR 1578): download, extract EDIs, keep only the FORGE footprint =="
get /tmp/forge_mt.zip "https://gdr.openei.org/files/1578/GDR%20upload%20SWUtah-MT.zip"
if [ -s /tmp/forge_mt.zip ]; then
  unzip -p /tmp/forge_mt.zip "swUT_EDI.zip" > /tmp/forge_edi.zip 2>/dev/null
  unzip -o -q /tmp/forge_edi.zip -d /tmp/forge_swedi 2>/dev/null
  for z in /tmp/forge_swedi/swUT_EDI/*.zip; do unzip -j -o -C -q "$z" "*.edi" -d "$M/mt/edi/" 2>/dev/null; done
  unzip -p /tmp/forge_mt.zip "FRG230523_Mdl12_resistivity.zip" > "$M/mt/resistivity_model.zip" 2>/dev/null
  unzip -p /tmp/forge_mt.zip "FRG230523_Mdl12_soundings.zip"   > "$M/mt/soundings.zip" 2>/dev/null
  unzip -p /tmp/forge_mt.zip "Wannamaker (2022).pdf"           > "$M/mt/Wannamaker_2022_MT_report.pdf" 2>/dev/null
  rm -rf /tmp/forge_mt.zip /tmp/forge_edi.zip /tmp/forge_swedi
  python3 - "$M/mt/edi" <<'PY'
import glob, re, os, sys
LAT0,LAT1,LON0,LON1 = 38.38, 38.62, -113.02, -112.76   # FORGE / Roosevelt Hot Springs
def dms(s):
    m=re.match(r'([+-]?)(\d+):(\d+):([\d.]+)', s.strip())
    if not m:
        try: return float(s)
        except: return None
    return (-1 if m.group(1)=='-' else 1)*(int(m.group(2))+int(m.group(3))/60+float(m.group(4))/3600)
d=sys.argv[1]; kept=0
for f in glob.glob(d+'/*.edi')+glob.glob(d+'/*.EDI'):
    t=open(f,errors='ignore').read(4000)
    la=re.search(r'REFLAT\s*=\s*([+\-0-9:.]+)',t) or re.search(r'\bLAT\s*=\s*([+\-0-9:.]+)',t)
    lo=re.search(r'REFLONG\s*=\s*([+\-0-9:.]+)',t) or re.search(r'\bLONG\s*=\s*([+\-0-9:.]+)',t)
    lat=dms(la.group(1)) if la else None; lon=dms(lo.group(1)) if lo else None
    if lat and lon and LAT0<=lat<=LAT1 and LON0<=lon<=LON1: kept+=1
    else: os.remove(f)
print(f"  kept {kept} MT sites in the FORGE footprint")
PY
fi

echo "== well logs: LAS (small, always) =="
get "$M/welllog/16A/16A-78-32_Spectral.las" "https://gdr.openei.org/files/1292/UnivUtah_Forge-16A-78-32_Spectral.las"
get "$M/welllog/58-32/58-32_DSI_Sonic.las"  "https://gdr.openei.org/files/1006/D5RL-00187_University%20of%20Utah_ME-ESW1_Run1_DSI%20Sonic.las"
get "$M/welllog/58-32/58-32_PT_temperature_logs.zip" "https://gdr.openei.org/files/1006/58-32_PT_logs.zip"

echo "== InSAR (GDR 1154) + microseismic helper (GDR 1207) =="
get "$M/microseismic/get_DAS_geophone_data.sh" "https://gdr.openei.org/files/1207/get_all_slb2%20(1).sh"
if [ "$SMALL" = "0" ]; then
  get "$M/insar/insar_2019.zip" "https://gdr.openei.org/files/1154/insar_for_GDR_feigl_20190701.zip"
  echo "== large well-log archives (~10 GB) =="
  get "$M/welllog/16A/16A_CBL.zip"               "https://gdr.openei.org/files/1292/16A(78)_32%20CBL%20Wireline.zip"
  get "$M/welllog/16A/16A_mudlog_temperature.pdf" "https://gdr.openei.org/files/1292/16A(78)-32%20Mud%20Log%20Final%20130-10987%27.pdf"
  get "$M/welllog/16A/16A_Sanvean.zip"           "https://gdr.openei.org/files/1292/Forge%2016A(78)-32%20Sanvean%20Log.zip"
  get "$M/welllog/16A/16A_Schlumberger.zip"      "https://gdr.openei.org/files/1292/16A(78)-32%20Schlumberger.zip"
  get "$M/welllog/58-32/58-32_EOWR.zip"          "https://gdr.openei.org/files/1006/58-32_EOWR.zip"
  get "$M/welllog/58-32/58-32_dipole_sonic.zip"  "https://gdr.openei.org/files/1006/Forge%2058-32%20Monitor%20well%20Dipole%20Sonic%20Data.zip"
  get "$M/welllog/58-32/58-32_logs.zip"          "https://gdr.openei.org/files/1006/58-32_logs.zip"
  echo "== extracting LAS files from the well-log archives =="
  for z in "$M"/welllog/16A/*.zip "$M"/welllog/58-32/*.zip; do
    [ -s "$z" ] && unzip -j -o -C -q "$z" "*.las" -d "$(dirname "$z")" 2>/dev/null
  done
fi

echo "== done. total =="; du -sh "$M"
