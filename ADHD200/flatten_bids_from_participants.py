#!/usr/bin/env python3

import argparse, json, os, re, shutil
from pathlib import Path
from typing import List
import pandas as pd

def slug(text: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', str(text).lower())

def parse_sites_field(val: str) -> List[str]:
    if val is None: return []
    s = str(val).strip()
    if not s: return []
    if s.startswith('[') and s.endswith(']'):
        s2 = s.strip('[]')
        parts = re.split(r'[,\|;]+', s2)
        return [p.strip().strip('\"').strip("'") for p in parts if p.strip()]
    parts = re.split(r'[,\|;]+', s)
    return [p.strip() for p in parts if p.strip()]

def discover_subject_paths(src_root: Path, site_names: List[str], scan_id: str) -> List[Path]:
    candidates = []
    subj_variants = [f"sub-{scan_id}", f"sub-{scan_id.zfill(3)}", f"sub-{scan_id.zfill(4)}", f"sub-{scan_id.zfill(5)}"]
    for site in site_names:
        site_dir = src_root / site
        if not site_dir.exists(): continue
        for v in subj_variants:
            p = site_dir / v
            if p.is_dir(): candidates.append(p)
    return candidates

def copy_or_link(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists(): return
    if mode == 'symlink':
        try:
            os.symlink(os.path.abspath(src), dst); return
        except OSError:
            pass
    shutil.copy2(src, dst)

def ensure_json_sidecar(dst_dir: Path, nii_file: Path, src_dir: Path):
    """Ensure a JSON sidecar exists for a given NIfTI file.

    Search order:
      1. A json with identical stem in the same directory as the NIfTI (specific)
      2. Generic candidates (task-rest_bold.json, bold.json, T1w.json, task-<task>_bold.json) in src_dir
      3. Same generic names in up to three ancestor directories above src_dir

    Returns (ok: bool, path or None, note: str)
    """
    import json, re
    base = nii_file.name
    json_name = re.sub(r'\.nii(\.gz)?$', '.json', base)
    dst_json = dst_dir / json_name
    if dst_json.exists():
        return True, dst_json, 'sidecar-already-present'

    # 1. Specific sidecar with matching basename
    src_specific = src_dir / json_name
    if src_specific.exists():
        try:
            data = json.loads(src_specific.read_text())
        except Exception:
            data = None
        if data is None:
            try:
                os.symlink(os.path.abspath(src_specific), dst_json)
            except OSError:
                shutil.copy2(src_specific, dst_json)
        else:
            dst_json.write_text(json.dumps(data, indent=2))
        return True, dst_json, 'specific-sidecar-copied'

    # 2/3. Generic candidates in current and ancestor dirs
    def build_generic_list(directory: Path):
        generic = []
        if '_bold' in base:
            # Extract task, acquisition number if any
            # Examples:
            #  sub-XXX_task-rest_acq-2_run-1_bold.nii.gz -> want task-rest_acq-2_bold.json first
            #  sub-XXX_task-rest_bold.nii.gz -> task-rest_bold.json
            m_task = re.search(r'(task-[a-zA-Z0-9]+)', base)
            m_acq  = re.search(r'(acq-[a-zA-Z0-9]+)', base)
            # Highest priority: task + acq specific
            if m_task and m_acq:
                generic.append(directory / f"{m_task.group(1)}_{m_acq.group(1)}_bold.json")
            # Next: any task-rest_acq-X_bold.json if acq present but maybe variant naming
            if m_acq and (not m_task or (m_task and m_task.group(1) != 'task-rest')):
                # still prefer task-rest with same acq if exists
                generic.append(directory / f"task-rest_{m_acq.group(1)}_bold.json")
            # Task-level without acquisition
            if m_task:
                generic.append(directory / f"{m_task.group(1)}_bold.json")
            # Acquisition-only (rare pattern)
            if m_acq:
                generic.append(directory / f"{m_acq.group(1)}_bold.json")
            # Fallbacks
            generic.append(directory / 'task-rest_bold.json')
            generic.append(directory / 'bold.json')
        if '_T1w' in base or base.endswith('T1w.nii') or base.endswith('T1w.nii.gz'):
            generic.append(directory / 'T1w.json')
        return generic

    # Directories to search: src_dir, parent, grandparent, great-grandparent
    search_dirs = []
    cur = src_dir
    for _ in range(4):  # current + 3 levels above
        if cur and cur not in search_dirs:
            search_dirs.append(cur)
        if cur.parent == cur:
            break
        cur = cur.parent

    for level, d in enumerate(search_dirs):
        candidates = build_generic_list(d)
        for gc in candidates:
            if gc.exists():
                try:
                    data = json.loads(gc.read_text())
                except Exception:
                    data = {}
                # Write JSON matching final nii base name
                dst_json.write_text(json.dumps(data, indent=2))
                origin = 'generic-current' if level == 0 else f'generic-up-{level}'
                return True, dst_json, f'cloned-from-{origin}:{gc.name}'

    return False, None, 'no-sidecar-found'

def sex_to_bids(val) -> str:
    if val is None: return 'n/a'
    s = str(val).strip().lower()
    if s in {'m','M','male','Male','1','1.0'}: return 'M'
    if s in {'f','F','female','Female','0','0.0'}: return 'F'
    return 'n/a'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--participants-csv', required=True)
    ap.add_argument('--source-root', required=True)
    ap.add_argument('--dest-root', required=True)
    ap.add_argument('--id-column', default='ScanDirID')
    ap.add_argument('--site-column', default='Site')
    ap.add_argument('--sites-column', default='Sites')
    ap.add_argument('--age-column', default='Age')
    ap.add_argument('--sex-column', default='Gender')
    ap.add_argument('--diagnosis-column', default='DX')
    ap.add_argument('--mode', choices=['symlink','copy'], default='symlink')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    src_root = Path(args.source_root).expanduser().resolve()
    dst_root = Path(args.dest_root).expanduser().resolve()
    dst_root.mkdir(parents=True, exist_ok=True)

    # Read participants CSV ensuring the ID column is treated as string to preserve leading zeros
    # (important for scan IDs like 0026001). If the user supplied an id column name variant
    # differing only by spaces, normalize it.
    df = pd.read_csv(args.participants_csv, dtype={args.id_column: str})
    if args.id_column not in df.columns:
        # Try to find a column whose name matches after removing spaces
        target_norm = args.id_column.replace(' ', '').lower()
        for col in list(df.columns):
            if col.replace(' ', '').lower() == target_norm:
                df.rename(columns={col: args.id_column}, inplace=True)
                break
    # Re-assert presence after attempted normalization
    for c in [args.id_column, args.site_column, args.sites_column, args.age_column, args.sex_column]:
        if c not in df.columns:
            raise SystemExit(f'Missing column in CSV: {c}')

    # dataset_description.json
    ds_path = dst_root / 'dataset_description.json'
    if not ds_path.exists() and not args.dry_run:
        ds_path.write_text(json.dumps({
            'Name': 'ADHD200_flattened',
            'BIDSVersion': '1.8.0',
            'DatasetType': 'raw'
        }, indent=2))

    report = []
    parts = []
    class_labels = []

    # Iterate rows (fixed: removed erroneous .strip() on iterator)
    for _, row in df.iterrows():
        # Preserve original string (already read as str); strip surrounding whitespace only
        raw_scan_id = str(row[args.id_column]).strip()
        scan_id = raw_scan_id  # alias for clarity
        site_val = row[args.site_column]
        age_val  = row[args.age_column]
        sex_val  = row[args.sex_column]
        diagnosis_val = row[args.diagnosis_column]
        sites_list = parse_sites_field(row[args.sites_column])
        site_candidates = [s for s in sites_list if s] or [str(site_val).strip()]

        site_code = slug(site_candidates[0]) if site_candidates else slug(str(site_val))
        # new_label = f"{site_code}{re.sub(r'[^0-9a-zA-Z]+','',scan_id)}".lower()
        new_label = f"{scan_id}".lower() # just use scan ID as is for new label

        # Discover subject paths; include original as-is plus zero-padded variants up to length of original
        subj_paths = discover_subject_paths(src_root, site_candidates, scan_id)
        if not subj_paths:
            report.append({'orig_scandir_id': scan_id,
                           'orig_site': site_val,
                           'sites_tried': '|'.join(site_candidates),
                           'status': 'not-found',
                           'dst_subject': f'sub-{new_label}'})
            continue

        src_sub = subj_paths[0]
        dst_sub = dst_root / f'sub-{new_label}'
        copied = 0
        missing_side = 0

        # Support either direct modality folders (anat/func/...) or session-level folders like ses-1/anat
        modality_names = ('anat','func','fmap','dwi')
        session_dirs = [d for d in src_sub.iterdir() if d.is_dir() and re.match(r'ses-[A-Za-z0-9]+', d.name)]

        def iter_modality_dirs():
            if session_dirs:
                for ses in session_dirs:
                    for mod in modality_names:
                        mdir = ses / mod
                        if mdir.is_dir():
                            yield mod, mdir
            # Also handle any direct modality dirs at subject root (even if sessions exist)
            for mod in modality_names:
                mdir = src_sub / mod
                if mdir.is_dir():
                    yield mod, mdir

        seen_any = False
        for mod, src_mod in iter_modality_dirs():
            seen_any = True
            for f in src_mod.rglob('*'):
                if not f.is_file():
                    continue
                if not f.name.lower().endswith(('.nii','.nii.gz','.json','.tsv','.tsv.gz')):
                    continue
                rel_parent = f.relative_to(src_sub).parent  # keeps session folder if present
                new_name = re.sub(r'sub-[0-9]+', f'sub-{new_label}', f.name)
                dst_path = dst_sub / rel_parent / new_name

                if args.dry_run:
                    report.append({'orig_scandir_id': scan_id,
                                   'orig_site': site_val,
                                   'sites_tried': '|'.join(site_candidates),
                                   'status': f'would-link-{mod}',
                                   'src': str(f),
                                   'dst': str(dst_path)})
                    continue

                if f.suffix in {'.json','.tsv'} or f.name.lower().endswith('.tsv.gz'):
                    copy_or_link(f, dst_path, args.mode)
                    copied += 1
                elif f.name.lower().endswith(('.nii','.nii.gz')):
                    copy_or_link(f, dst_path, args.mode)
                    copied += 1
                    ok, dst_json_path, note = ensure_json_sidecar(dst_path.parent, dst_path, f.parent)
                    if not ok:
                        missing_side += 1
                        report.append({'orig_scandir_id': scan_id,
                                       'orig_site': site_val,
                                       'sites_tried': '|'.join(site_candidates),
                                       'status': 'missing-sidecar',
                                       'nii': str(dst_path)})

        if not seen_any:
            # record absence of modalities
            report.append({'orig_scandir_id': scan_id,
                           'orig_site': site_val,
                           'sites_tried': '|'.join(site_candidates),
                           'status': 'no-modalities-found',
                           'dst_subject': f'sub-{new_label}',
                           'src_subject': str(src_sub)})

        report.append({'orig_scandir_id': scan_id,
                       'orig_site': site_val,
                       'sites_tried': '|'.join(site_candidates),
                       'status': f'ok:copied={copied};missing_sidecars={missing_side}',
                       'dst_subject': f'sub-{new_label}',
                       'src_subject': str(src_sub)})

        parts.append({'participant_id': f'sub-{new_label}',
                      'age': age_val if pd.notna(age_val) else 'n/a',
                      'gender': sex_to_bids(sex_val),
                      'diagnosis': diagnosis_val if pd.notna(diagnosis_val) else 'n/a',
                      'site': site_candidates[0] if site_candidates else str(site_val),
                      'orig_site': site_val,
                      'orig_scandir_id': scan_id,
                      'orig_scan_id_preserved': raw_scan_id})

        class_labels.append({'participant_id': f'sub-{new_label}',
                             'label': diagnosis_val if pd.notna(diagnosis_val) else 'n/a'})

    # write report, participants.tsv and participants_class_labels.tsv
    pd.DataFrame(report).to_csv(dst_root / 'flatten_report.tsv', sep='\t', index=False)

    p_path = dst_root / 'participants.tsv'
    new_df = pd.DataFrame(parts)
    if p_path.exists():
        try:
            old = pd.read_csv(p_path, sep='\t')
        except Exception:
            old = pd.DataFrame()
        pd.concat([old, new_df], ignore_index=True).drop_duplicates(subset=['participant_id'], keep='last').to_csv(p_path, sep='\t', index=False)
    else:
        new_df.to_csv(p_path, sep='\t', index=False)

    l_path = dst_root / 'participants_class_labels.tsv'
    new_df = pd.DataFrame(class_labels)
    if l_path.exists():
        try:
            old = pd.read_csv(l_path, sep='\t')
        except Exception:
            old = pd.DataFrame()
            pd.concat([old, new_df], ignore_index=True).drop_duplicates(subset=['participant_id'], keep='last').to_csv(l_path, sep='\t', index=False)
    else:
        new_df.to_csv(l_path, sep='\t', index=False)

    print('Done. See:', dst_root / 'flatten_report.tsv', 'and', p_path)

if __name__ == '__main__':
    main()
