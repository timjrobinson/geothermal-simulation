# Utah FORGE — real multi-method test dataset

A **co-located, real-world** geophysical dataset for the Utah FORGE site (the U.S. DOE
Enhanced Geothermal Systems field laboratory near **Milford / Roosevelt Hot Springs,
Utah**) — the real-world analogue of the synthetic `great-basin-v1` scenario. Every
dataset here sits over the **same ~30 × 30 km footprint** (centred on the FORGE well pad,
≈ 38.50 °N, −112.89 °W), so the methods can be fused into one earth model.

All data is published openly on the **[Geothermal Data Repository (GDR)](https://gdr.openei.org)**
(U.S. DOE Geothermal Technologies Office) under CC-BY 4.0. This folder ships only the
*recipe*; run [`fetch.sh`](fetch.sh) to download the actual files (they are large and are
git-ignored).

## What's included (all co-located at FORGE)

| Method | What | Format | GDR source |
|---|---|---|---|
| **Gravity** | ~1000 Bouguer-anomaly stations (UGS + PACES) | CSV/text | [1002](https://gdr.openei.org/submissions/1002) |
| **EM / TEM** | 68 transient-EM soundings | USF | [1002](https://gdr.openei.org/submissions/1002) |
| **Magnetotellurics** | 113 MT sites (impedance tensors) over FORGE + Roosevelt Hot Springs, filtered from the SW-Utah survey by coordinate, **plus the FORGE 3-D inverted resistivity model** | EDI + model | [1578](https://gdr.openei.org/submissions/1578) |
| **Well logs** | Deep wells **16A(78)-32** and **58-32**: sonic (Vp/Vs), resistivity, gamma, density, image logs, anisotropy | LAS + archives | [1292](https://gdr.openei.org/submissions/1292), [1006](https://gdr.openei.org/submissions/1006) |
| **Temperature** | Pressure/temperature logs and mud-log temperatures from the deep wells | LAS/PDF | [1006](https://gdr.openei.org/submissions/1006), [1292](https://gdr.openei.org/submissions/1292) |
| **InSAR** | Ground-deformation rate (mean range-change + σ), 2019 | CSV/NetCDF | [1154](https://gdr.openei.org/submissions/1154) |
| **Microseismic** | Helper script to pull the continuous geophone/DAS waveforms (hosted at U. Utah CHPC) | shell + docs | [1207](https://gdr.openei.org/submissions/1207) |

**Not readily available as a co-located open product** (the synthetic generator covers
these): a dedicated FORGE **aeromagnetic** survey and an active-source **3-D seismic
reflection** cube. Continuous microseismic waveforms are tens of GB and live off-site
(fetch via `measured/microseismic/get_DAS_geophone_data.sh`).

## Folder layout

```
data/utah-forge/
  frame.json     # the project SpatialFrame (doc 01): UTM 12N (EPSG:32612), anchored at FORGE
  manifest.json  # per-file inventory + provenance (DOIs, URLs)
  fetch.sh       # reproducible downloader (run this to populate measured/)
  measured/      # the downloaded native files (git-ignored — large)
    gravity/  em/  mt/  welllog/{16A,58-32}/  insar/  microseismic/
```

## Download it

```bash
cd data/utah-forge
./fetch.sh           # small files + MT (coordinate-filtered) + the large well/InSAR archives
./fetch.sh --small   # skip the multi-GB well-log archives
```

## Load it into the simulator

```bash
# from the repo root, with the backend venv (make setup) ready:
backend/.venv/bin/python data/load_utah_forge.py --storage-root .devdata-forge
```

This creates a georeferenced project from `frame.json` and ingests every native file
(`.edi`, `.las`, `.usf`, gravity CSV, InSAR CSV) through the normal ingestion pipeline,
reprojecting each into the FORGE Engineering Frame. Then point the viewer at it
(`make run-frontend`).

### Ingestion status (real-world hardening)

The adapters were first built against the synthetic generator's outputs; real field files
carry vendor quirks, so this dataset is also a hardening target. Current status:

| Method | Files | Ingests? |
|---|---|---|
| **Gravity** | 1 (3735 stations) | ✅ — adapter now reads `gCBGA/gSBGA/gFA` + lon/lat |
| **Magnetotellurics** | 113 sites | ✅ — adapter now computes apparent resistivity + phase from the impedance tensor (`>ZXYR/>ZXYI`) and reads `REFLAT/REFLONG` |
| **EM / TEM** | 68 soundings | ⚠️ — Zonge `.usf` format not yet parsed (synthetic EM uses a different layout) |
| **Well logs / temperature** | LAS + archives | ⚠️ — MD-indexed curves need CRS-free placement; adapter tweak pending |
| **InSAR** | CSV/NetCDF (in zip) | ⚠️ — extract from `insar_2019.zip`; column mapping pending |

So **gravity + 113 MT sites load today** (enough to build a fused resistivity + density
model and invert). The ⚠️ methods are concrete next adapter-hardening steps. The loader
prints per-file success/failure.

## Citation

Data © their respective providers (Energy & Geoscience Institute / University of Utah,
Utah Geological Survey, et al.), via the DOE Geothermal Data Repository, CC-BY 4.0.
See `manifest.json` for per-dataset DOIs.
