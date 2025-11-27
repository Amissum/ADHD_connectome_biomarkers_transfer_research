#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ADHD200 derivatives → ROI time series (for LSTM / DL pipelines)

This script walks fMRIPrep derivatives, pairs each *desc-preproc_bold.nii.gz with its
confounds_timeseries.tsv and brain_mask, applies confound regression + scrubbing, and
extracts ROI time series using a labels atlas (e.g., Schaefer). It then saves per-subject
timeseries as TSV/NPY and aggregates them into a single NPZ for modeling.

Requirements (install in your environment):
- numpy, pandas, nibabel, nilearn, scipy

Usage (example):
python adhd200_derivs_pipeline.py \
  --deriv-root /path/to/derivatives/fmriprep \
  --atlas-img   /path/to/atlas/Schaefer2018_400Parcels_17Networks_order_FSLMNI152_2mm.nii.gz \
  --atlas-labels /path/to/atlas/Schaefer2018_400Parcels_17Networks_order.txt \
  --participants /path/to/cohort_12_15_ALL_ids.txt \
  --out /path/to/derivatives_timeseries/cohort_12_15 \
  --fd-thresh 0.5 \
  --low-pass 0.08 \
  --high-pass 0.008 \
  --n-acompcor 5 \
  --use-gsr 0 \
  --min-vols 120
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import nibabel as nib

from nilearn.maskers import NiftiLabelsMasker
from nilearn.signal import clean as nilearn_clean


def find_func_derivs(deriv_root: Path, subjects: Optional[List[str]] = None) -> List[Dict[str, Path]]:
    patterns = sorted(deriv_root.glob("sub-*/*/func/*desc-preproc_bold.nii.gz"))
    if not patterns:
        patterns = sorted(deriv_root.glob("sub-*/func/*desc-preproc_bold.nii.gz"))
    out = []
    for bold in patterns:
        parts = bold.name.split("_")
        info = {"bold": bold, "confounds": None, "mask": None, "sub": None, "ses": None, "task": None, "run": None}
        for p in parts:
            if p.startswith("sub-"):
                info["sub"] = p[4:]
            elif p.startswith("ses-"):
                info["ses"] = p[4:]
            elif p.startswith("task-"):
                info["task"] = p[5:]
            elif p.startswith("run-"):
                info["run"] = p[4:]
        if subjects and info["sub"] not in subjects:
            continue
        conf = bold.parent / bold.name.replace("_desc-preproc_bold.nii.gz", "_desc-confounds_timeseries.tsv")
        mask = bold.parent / bold.name.replace("_desc-preproc_bold.nii.gz", "_desc-brain_mask.nii.gz")
        if conf.exists():
            info["confounds"] = conf
        if mask.exists():
            info["mask"] = mask
        out.append(info)
    return out


def load_confounds(conf_path: Path, fd_thresh: float = 0.5, n_acompcor: int = 5, use_gsr: bool = False,
                   add_cols: Optional[List[str]] = None) -> Tuple[np.ndarray, Optional[np.ndarray], Dict]:
    cf = pd.read_csv(conf_path, sep="\t")
    meta: Dict = {}
    fd = cf.get("framewise_displacement")
    sample_mask = None
    if fd is not None:
        bad = fd.fillna(0).values > fd_thresh
        nss_cols = [c for c in cf.columns if c.startswith("non_steady_state_outlier")]
        if nss_cols:
            nss = (cf[nss_cols].fillna(0).sum(axis=1) > 0).values
            bad = np.logical_or(bad, nss)
        good_idx = np.where(~bad)[0]
        if len(good_idx) > 0:
            sample_mask = good_idx
        meta["n_scrubbed"] = int(bad.sum())
    else:
        meta["n_scrubbed"] = 0

    cols = []
    motion_cols = [c for c in ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"] if c in cf.columns]
    motion_derivs = [c + "_derivative1" for c in motion_cols if c + "_derivative1" in cf.columns]
    cols += motion_cols + motion_derivs
    acomp = [c for c in cf.columns if c.startswith("a_comp_cor")]
    acomp = acomp[:n_acompcor] if n_acompcor > 0 else []
    cols += acomp
    for c in ["white_matter", "csf"]:
        if c in cf.columns:
            cols.append(c)
    if use_gsr and "global_signal" in cf.columns:
        cols.append("global_signal")
    if add_cols:
        cols += [c for c in add_cols if c in cf.columns]
    X = cf[cols].fillna(0).values if cols else None
    meta["confound_cols"] = cols
    return X, sample_mask, meta


def extract_roi_timeseries(bold_img: Path, atlas_img: Path, brain_mask: Optional[Path], confounds: Optional[np.ndarray],
                           sample_mask: Optional[np.ndarray], tr: Optional[float], low_pass: Optional[float],
                           high_pass: Optional[float], standardize: str = "zscore", detrend: bool = True) -> np.ndarray:
    masker = NiftiLabelsMasker(
        labels_img=str(atlas_img),
        mask_img=str(brain_mask) if brain_mask else None,
        standardize=standardize,
        detrend=detrend,
        high_pass=high_pass,
        low_pass=low_pass,
        t_r=tr,
        resampling_target="data",
        strategy=None,
    )
    ts = masker.fit_transform(str(bold_img), confounds=confounds, sample_mask=sample_mask)
    return ts


def guess_tr(bold_img: Path) -> Optional[float]:
    img = nib.load(str(bold_img))
    hdr = img.header
    try:
        tr = float(hdr.get_zooms()[3])
        if np.isfinite(tr) and tr > 0:
            return tr
    except Exception:
        pass
    return None


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--deriv-root", required=True)
    parser.add_argument("--atlas-img", required=True)
    parser.add_argument("--atlas-labels", default=None)
    parser.add_argument("--participants", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--fd-thresh", type=float, default=0.5)
    parser.add_argument("--n-acompcor", type=int, default=5)
    parser.add_argument("--use-gsr", type=int, default=0)
    parser.add_argument("--low-pass", type=float, default=0.08)
    parser.add_argument("--high-pass", type=float, default=0.008)
    parser.add_argument("--min-vols", type=int, default=120)
    args = parser.parse_args()

    deriv_root = Path(args.deriv_root).expanduser().resolve()
    atlas_img = Path(args.atlas_img).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    subjects = None
    if args.participants:
        with open(args.participants, "r") as f:
            subjects = [ln.strip().replace("sub-", "") for ln in f if ln.strip()]

    runs = find_func_derivs(deriv_root, subjects)
    print(f"Found {len(runs)} BOLD runs.")

    roi_labels = None
    if args.atlas_labels and Path(args.atlas_labels).exists():
        lab_path = Path(args.atlas_labels)
        if lab_path.suffix.lower() == ".tsv":
            lab_df = pd.read_csv(lab_path, sep="\t")
            roi_labels = lab_df.get("label") or lab_df.get("name")
            if roi_labels is not None:
                roi_labels = [str(x) for x in roi_labels.tolist()]
        else:
            with open(lab_path, "r") as f:
                roi_labels = [ln.strip() for ln in f if ln.strip()]

    all_X = []
    meta_rows = []

    for r in runs:
        sub, ses, run, task = r["sub"], r["ses"], r["run"], r["task"]
        bold, conf, mask = r["bold"], r["confounds"], r["mask"]

        conf_mtx, sample_mask, conf_meta = load_confounds(
            conf, fd_thresh=args.fd_thresh, n_acompcor=args.n_acompcor, use_gsr=bool(args.use_gsr)
        )
        tr = guess_tr(bold)

        ts = extract_roi_timeseries(
            bold_img=bold, atlas_img=atlas_img, brain_mask=mask,
            confounds=conf_mtx, sample_mask=sample_mask,
            tr=tr, low_pass=args.low_pass, high_pass=args.high_pass,
            standardize="zscore", detrend=True,
        )

        if ts.shape[0] < args.min_vols:
            keep = False
        else:
            keep = True

        sub_dir = out_dir / f"sub-{sub}" / (f"ses-{ses}" if ses else "ses-NA")
        sub_dir.mkdir(parents=True, exist_ok=True)
        tsv_name = f"sub-{sub}_{('ses-'+ses+'_') if ses else ''}task-{task or 'rest'}_{('run-'+run+'_') if run else ''}timeseries.tsv"
        npy_name = tsv_name.replace(".tsv", ".npy")
        tsv_path = sub_dir / tsv_name
        npy_path = sub_dir / npy_name

        cols = [f"roi_{k+1}" for k in range(ts.shape[1])] if roi_labels is None else roi_labels[:ts.shape[1]]
        pd.DataFrame(ts, columns=cols).to_csv(tsv_path, sep="\t", index=False)
        np.save(npy_path, ts)

        meta_rows.append({
            "subject": sub, "session": ses or "", "run": run or "", "task": task or "rest",
            "bold": str(bold), "confounds": str(conf), "mask": str(mask) if mask else "",
            "tr": tr, "n_volumes_after_scrub": int(ts.shape[0]), "kept": int(keep),
            "n_confounds": int(0 if conf_mtx is None else conf_mtx.shape[1]),
            "n_scrubbed": int(conf_meta.get("n_scrubbed", 0)),
        })
        if keep:
            all_X.append(ts)

    agg_npz = out_dir / "timeseries_agg.npz"
    np.savez(agg_npz, *all_X)
    pd.DataFrame(meta_rows).to_csv(out_dir / "processing_manifest.tsv", sep="\t", index=False)
    print("Done. Outputs saved under:", out_dir)


if __name__ == "__main__":
    main()
