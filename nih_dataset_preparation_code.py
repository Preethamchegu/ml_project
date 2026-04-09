# ==============================================================================
# NIH ChestX-ray14 — Dataset Preparation Script
# Bacterial vs Viral Pathogen Filtering + Train/Val/Test Split
#
# HOW TO USE:
#   1. Set NIH_ROOT to the folder containing your NIH dataset files.
#   2. Set OUT_DIR to wherever you want the output CSVs saved.
#   3. Run:  python nih_dataset_preparation.py
#
# INPUT  (set paths below):
#   NIH_ROOT/
#     ├── Data_Entry_2017.csv
#     ├── train_val_list.txt
#     ├── test_list.txt
#     └── images_001/ ... images_012/   ← folders of .png files
#
# OUTPUT (saved to OUT_DIR):
#   train.csv  |  val.csv  |  test.csv
#   dataset_stats.txt
#   dataset_overview.png
#   samples_HEALTHY.png  |  samples_BACTERIAL.png
#   samples_VIRAL.png    |  samples_CO-INFECTION.png
# ==============================================================================

# ── USER-CONFIGURABLE PATHS ───────────────────────────────────────────────────

NIH_ROOT = '/home/i24ai006/datasets/nih'              # ← folder with Data_Entry_2017.csv etc.
OUT_DIR  = '/home/i24ai006/datasets/prepared'         # ← outputs saved here

# ─────────────────────────────────────────────────────────────────────────────


# ==============================================================================
# CELL 1: IMPORTS
# ==============================================================================
import os
import glob
import random
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit

warnings.filterwarnings('ignore')
random.seed(42)
np.random.seed(42)

os.makedirs(OUT_DIR, exist_ok=True)
print('✅ Imports ready')


# ==============================================================================
# CELL 2: NIH DATASET PATHS
# ==============================================================================

NIH_IMG_DIRS   = sorted(glob.glob(os.path.join(NIH_ROOT, 'images_*')))
NIH_CSV        = os.path.join(NIH_ROOT, 'Data_Entry_2017.csv')
NIH_TRAIN_LIST = os.path.join(NIH_ROOT, 'train_val_list.txt')
NIH_TEST_LIST  = os.path.join(NIH_ROOT, 'test_list.txt')

print('📂 Checking NIH dataset files...')
for name, path in [
    ('Data_Entry_2017.csv', NIH_CSV),
    ('train_val_list.txt',  NIH_TRAIN_LIST),
    ('test_list.txt',       NIH_TEST_LIST),
]:
    status = '✅' if os.path.exists(path) else '❌ MISSING'
    print(f'  {status}  {name}')

print(f'\n📁 Image sub-folders found: {len(NIH_IMG_DIRS)}')
for d in NIH_IMG_DIRS:
    imgs = glob.glob(os.path.join(d, '*.png'))
    print(f'   {os.path.basename(d)}: {len(imgs):,} images')


# ==============================================================================
# CELL 3: PATHOGEN LABEL MAPPING
#
# Out of 14 NIH labels, only these are caused by bacterial or viral infection
# and are visible on chest X-ray as infectious patterns.
#
# EXCLUDED: Emphysema, Fibrosis, Cardiomegaly, Nodule, Mass,
#           Hernia, Pleural_Thickening, Pneumothorax
# These are structural/chronic/mechanical — not infectious.
# ==============================================================================

# Labels that indicate BACTERIAL infection
BACTERIAL_LABELS = [
    'Consolidation',   # lobar dense filling — Streptococcus, Klebsiella, H.influenzae
    'Pneumonia',       # generic pneumonia label (shared, bacterial dominant)
    'Atelectasis',     # collapse — secondary to bacterial mucus plugging
    'Infiltration',    # patchy bronchopneumonia — gram-negative organisms
]

# Labels that indicate VIRAL infection
VIRAL_LABELS = [
    'Infiltration',    # bilateral interstitial infiltrates — influenza, RSV, COVID
    'Pneumonia',       # viral pneumonia (shared with bacterial)
    'Effusion',        # pleural effusion — parapneumonic / viral pleuritis
    'Edema',           # pulmonary edema — viral alveolar damage (ARDS pattern)
]

EXCLUDED_LABELS = [
    'Emphysema', 'Fibrosis', 'Cardiomegaly', 'Nodule',
    'Mass', 'Hernia', 'Pleural_Thickening', 'Pneumothorax',
]

ALL_PATHOGEN_LABELS = sorted(set(BACTERIAL_LABELS + VIRAL_LABELS))

print('🦠 BACTERIAL labels:', BACTERIAL_LABELS)
print('🦠 VIRAL labels    :', VIRAL_LABELS)
print('🔗 SHARED labels   :', sorted(set(BACTERIAL_LABELS) & set(VIRAL_LABELS)))
print('🚫 EXCLUDED labels :', EXCLUDED_LABELS)
print()
print('📋 All pathogen labels we keep:', ALL_PATHOGEN_LABELS)
print(f'   → {len(ALL_PATHOGEN_LABELS)} labels total (out of 14)')


# ==============================================================================
# CELL 4: LOAD NIH CSV AND BUILD IMAGE PATH MAP
# ==============================================================================

print('📂 Loading Data_Entry_2017.csv ...')
df_raw = pd.read_csv(NIH_CSV)
print(f'   Rows: {len(df_raw):,}')
print(f'   Columns: {list(df_raw.columns)}')

# Build filename → full path map
print('\n🔍 Building image path index...')
path_map = {}
for img_dir in NIH_IMG_DIRS:
    for fpath in glob.glob(os.path.join(img_dir, '*.png')):
        path_map[os.path.basename(fpath)] = fpath
print(f'   Found {len(path_map):,} images across all sub-folders')

# Rename columns for clarity
df_raw = df_raw.rename(columns={
    'Image Index'    : 'image_file',
    'Finding Labels' : 'label_str',
    'Patient ID'     : 'patient_id',
    'Patient Age'    : 'age',
    'Patient Gender' : 'gender',
    'View Position'  : 'view',
})

# Attach full image path
df_raw['image_path'] = df_raw['image_file'].map(path_map)
missing = df_raw['image_path'].isna().sum()
print(f'   Images matched: {(~df_raw.image_path.isna()).sum():,}')
if missing > 0:
    print(f'   ⚠️  {missing} images not found on disk (sub-folders may be missing)')

df_raw = df_raw[~df_raw['image_path'].isna()].reset_index(drop=True)
print(f'   Working with: {len(df_raw):,} rows after path filter')


# ==============================================================================
# CELL 5: PARSE LABELS — FILTER TO PATHOGEN-RELEVANT ONLY
# ==============================================================================

def parse_raw_labels(label_str):
    """Parse pipe-separated string into list. 'No Finding' → []"""
    parts = [l.strip() for l in str(label_str).split('|')]
    return [] if parts == ['No Finding'] else parts

def get_bacterial(raw_labels):
    return [l for l in raw_labels if l in BACTERIAL_LABELS]

def get_viral(raw_labels):
    return [l for l in raw_labels if l in VIRAL_LABELS]

df_raw['raw_labels']       = df_raw['label_str'].apply(parse_raw_labels)
df_raw['bacterial_labels'] = df_raw['raw_labels'].apply(get_bacterial)
df_raw['viral_labels']     = df_raw['raw_labels'].apply(get_viral)
df_raw['has_bacterial']    = df_raw['bacterial_labels'].apply(len) > 0
df_raw['has_viral']        = df_raw['viral_labels'].apply(len) > 0
df_raw['is_coinfection']   = df_raw['has_bacterial'] & df_raw['has_viral']
df_raw['is_healthy']       = df_raw['raw_labels'].apply(len) == 0
df_raw['is_excluded_only'] = (
    ~df_raw['has_bacterial'] &
    ~df_raw['has_viral'] &
    ~df_raw['is_healthy']
)

def cascade_label(row):
    if row['is_healthy']:      return 'HEALTHY'
    if row['is_coinfection']:  return 'CO-INFECTION'
    if row['has_bacterial']:   return 'BACTERIAL'
    if row['has_viral']:       return 'VIRAL'
    return 'EXCLUDED'

df_raw['cascade_label'] = df_raw.apply(cascade_label, axis=1)

print('📊 Full dataset label distribution (before split):')
print()
for lbl, cnt in df_raw['cascade_label'].value_counts().items():
    pct = cnt / len(df_raw) * 100
    bar = '█' * int(pct / 2)
    print(f'  {lbl:<15}: {cnt:7,}  ({pct:5.1f}%)  {bar}')

print()
print('  Breakdown inside CO-INFECTION:')
coinf = df_raw[df_raw['is_coinfection']]
for b in BACTERIAL_LABELS:
    for v in VIRAL_LABELS:
        if b == v:
            continue
        cnt = int(coinf.apply(
            lambda r: b in r['bacterial_labels'] and v in r['viral_labels'], axis=1
        ).sum())
        if cnt > 0:
            print(f'    {b} + {v}: {cnt:,}')


# ==============================================================================
# CELL 6: FILTER — KEEP ONLY RELEVANT IMAGES
#
#  Keep:  HEALTHY, BACTERIAL, VIRAL, CO-INFECTION
#  Drop:  EXCLUDED — images that only have structural/non-infectious labels
# ==============================================================================

df = df_raw[df_raw['cascade_label'] != 'EXCLUDED'].copy().reset_index(drop=True)

print(f'Before filter : {len(df_raw):,} images')
print(f'After filter  : {len(df):,} images  (dropped {len(df_raw)-len(df):,} excluded-only)')
print()
print('Kept label distribution:')
for lbl, cnt in df['cascade_label'].value_counts().items():
    pct = cnt / len(df) * 100
    print(f'  {lbl:<15}: {cnt:7,}  ({pct:5.1f}%)')

print()
print('Unique patients in filtered set:', df['patient_id'].nunique())


# ==============================================================================
# CELL 7: PATIENT-LEVEL TRAIN / VAL / TEST SPLIT
#
#  RULE: One patient's images must ALL be in the same split (no leakage).
#  Uses official NIH train_val_list.txt / test_list.txt when available,
#  otherwise falls back to GroupShuffleSplit on patient_id.
#  Final split: Train 70% | Val 10% | Test 20%
# ==============================================================================

if os.path.exists(NIH_TRAIN_LIST) and os.path.exists(NIH_TEST_LIST):
    print('✅ Using official NIH train/test split files')

    with open(NIH_TRAIN_LIST) as f:
        nih_train_files = set(l.strip() for l in f)
    with open(NIH_TEST_LIST) as f:
        nih_test_files = set(l.strip() for l in f)

    df_test = df[df['image_file'].isin(nih_test_files)].copy()
    df_tv   = df[df['image_file'].isin(nih_train_files)].copy()

    # Split df_tv into train (87.5%) and val (12.5%) → gives 70/10/20 overall
    gss = GroupShuffleSplit(n_splits=1, test_size=0.125, random_state=42)
    train_idx, val_idx = next(gss.split(df_tv, groups=df_tv['patient_id']))
    df_train = df_tv.iloc[train_idx].copy().reset_index(drop=True)
    df_val   = df_tv.iloc[val_idx].copy().reset_index(drop=True)
    df_test  = df_test.reset_index(drop=True)

else:
    print('⚠️  NIH split files not found — using GroupShuffleSplit fallback')

    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    tv_idx, test_idx = next(gss1.split(df, groups=df['patient_id']))
    df_tv   = df.iloc[tv_idx].copy().reset_index(drop=True)
    df_test = df.iloc[test_idx].copy().reset_index(drop=True)

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.125, random_state=42)
    train_idx, val_idx = next(gss2.split(df_tv, groups=df_tv['patient_id']))
    df_train = df_tv.iloc[train_idx].copy().reset_index(drop=True)
    df_val   = df_tv.iloc[val_idx].copy().reset_index(drop=True)

# Print split stats
total = len(df_train) + len(df_val) + len(df_test)
print()
print(f'{"Split":<8} {"Images":>8} {"Patients":>9} {"% of total":>11}')
print('─' * 42)
for name, dset in [('Train', df_train), ('Val', df_val), ('Test', df_test)]:
    pct = len(dset) / total * 100
    print(f'{name:<8} {len(dset):>8,} {dset.patient_id.nunique():>9,} {pct:>10.1f}%')
print('─' * 42)
print(f'{"Total":<8} {total:>8,} {df.patient_id.nunique():>9,}')

# Verify no patient leakage
train_pts = set(df_train['patient_id'])
val_pts   = set(df_val['patient_id'])
test_pts  = set(df_test['patient_id'])
assert len(train_pts & val_pts)  == 0, 'LEAKAGE: train/val share patients!'
assert len(train_pts & test_pts) == 0, 'LEAKAGE: train/test share patients!'
assert len(val_pts   & test_pts) == 0, 'LEAKAGE: val/test share patients!'
print()
print('✅ No patient leakage between splits')


# ==============================================================================
# CELL 8: PER-SPLIT LABEL DISTRIBUTION CHECK
# ==============================================================================

def split_stats(dset, name):
    print(f'\n  ── {name} ({len(dset):,} images) ──')
    for lbl in ['HEALTHY', 'BACTERIAL', 'VIRAL', 'CO-INFECTION']:
        cnt = int((dset['cascade_label'] == lbl).sum())
        pct = cnt / len(dset) * 100
        bar = '█' * int(pct / 2)
        print(f'    {lbl:<15}: {cnt:6,}  ({pct:5.1f}%)  {bar}')
    print(f'    Unique patients: {dset.patient_id.nunique():,}')

print('📊 Label distribution per split:')
split_stats(df_train, 'TRAIN')
split_stats(df_val,   'VAL')
split_stats(df_test,  'TEST')

print('\n  ── Co-infection breakdown in TRAIN ──')
ci_train = df_train[df_train['cascade_label'] == 'CO-INFECTION']
pair_counts = {}
for _, row in ci_train.iterrows():
    for b in row['bacterial_labels']:
        for v in row['viral_labels']:
            if b != v:
                pair_counts[(b, v)] = pair_counts.get((b, v), 0) + 1
for (b, v), cnt in sorted(pair_counts.items(), key=lambda x: -x[1])[:8]:
    print(f'    {b} + {v}: {cnt:,}')


# ==============================================================================
# CELL 9: VISUALISATION — DATASET OVERVIEW
# ==============================================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('NIH ChestX-ray14 — Filtered Pathogen Dataset Overview',
             fontsize=14, fontweight='bold')

COLORS = {
    'HEALTHY':      '#4CAF50',
    'BACTERIAL':    '#FF9800',
    'VIRAL':        '#2196F3',
    'CO-INFECTION': '#9C27B0',
    'EXCLUDED':     '#9E9E9E',
}

# 1) Overall cascade label distribution
ax = axes[0, 0]
vc = df['cascade_label'].value_counts()
bars = ax.bar(vc.index, vc.values,
              color=[COLORS.get(l, '#ccc') for l in vc.index],
              edgecolor='white', linewidth=0.8)
ax.set_title('Overall cascade label distribution', fontweight='bold')
ax.set_ylabel('Count')
for bar, cnt in zip(bars, vc.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
            f'{cnt:,}', ha='center', fontsize=9, fontweight='bold')
ax.tick_params(axis='x', rotation=15)

# 2) Per-split label distribution
ax = axes[0, 1]
splits = ['Train', 'Val', 'Test']
dsets  = [df_train, df_val, df_test]
labels = ['HEALTHY', 'BACTERIAL', 'VIRAL', 'CO-INFECTION']
x = np.arange(len(splits))
w = 0.2
for i, lbl in enumerate(labels):
    counts = [int((d['cascade_label'] == lbl).sum()) for d in dsets]
    ax.bar(x + i * w, counts, w, label=lbl, color=COLORS[lbl],
           edgecolor='white', linewidth=0.5)
ax.set_title('Label distribution by split', fontweight='bold')
ax.set_xticks(x + w * 1.5)
ax.set_xticklabels(splits)
ax.set_ylabel('Count')
ax.legend(fontsize=8)

# 3) Bacterial label frequency
ax = axes[0, 2]
b_counts = {lbl: int(df['bacterial_labels'].apply(lambda l: lbl in l).sum())
            for lbl in BACTERIAL_LABELS}
bars = ax.barh(list(b_counts.keys()), list(b_counts.values()),
               color='#FF9800', edgecolor='white')
ax.set_title('Bacterial label frequency', fontweight='bold')
ax.set_xlabel('Count')
for bar, v in zip(bars, b_counts.values()):
    ax.text(v + 20, bar.get_y() + bar.get_height() / 2,
            f'{v:,}', va='center', fontsize=9)

# 4) Viral label frequency
ax = axes[1, 0]
v_counts = {lbl: int(df['viral_labels'].apply(lambda l: lbl in l).sum())
            for lbl in VIRAL_LABELS}
bars = ax.barh(list(v_counts.keys()), list(v_counts.values()),
               color='#2196F3', edgecolor='white')
ax.set_title('Viral label frequency', fontweight='bold')
ax.set_xlabel('Count')
for bar, v in zip(bars, v_counts.values()):
    ax.text(v + 20, bar.get_y() + bar.get_height() / 2,
            f'{v:,}', va='center', fontsize=9)

# 5) Co-infection pair heatmap
ax = axes[1, 1]
mat = np.zeros((len(BACTERIAL_LABELS), len(VIRAL_LABELS)))
ci  = df[df['cascade_label'] == 'CO-INFECTION']
for i, b in enumerate(BACTERIAL_LABELS):
    for j, v in enumerate(VIRAL_LABELS):
        mat[i, j] = int(ci.apply(
            lambda r: b in r['bacterial_labels'] and v in r['viral_labels'],
            axis=1).sum())
sns.heatmap(mat, annot=True, fmt='.0f', cmap='Purples',
            xticklabels=VIRAL_LABELS, yticklabels=BACTERIAL_LABELS, ax=ax,
            linewidths=0.5, cbar_kws={'shrink': 0.8})
ax.set_title('Co-infection pair counts', fontweight='bold')
ax.set_xlabel('Viral label')
ax.set_ylabel('Bacterial label')
ax.tick_params(axis='x', rotation=20)
ax.tick_params(axis='y', rotation=0)

# 6) Patient count per split
ax = axes[1, 2]
pt_counts = [d['patient_id'].nunique() for d in dsets]
bars = ax.bar(splits, pt_counts,
              color=['#2196F3', '#FF9800', '#4CAF50'],
              edgecolor='white')
ax.set_title('Unique patients per split', fontweight='bold')
ax.set_ylabel('Patients')
for bar, v in zip(bars, pt_counts):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
            f'{v:,}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'dataset_overview.png'), dpi=130, bbox_inches='tight')
plt.show()
print('✅ Visualisation saved')


# ==============================================================================
# CELL 10: SAVE TRAIN / VAL / TEST CSV FILES
#
# Each CSV columns:
#   image_file, image_path, patient_id, age, gender, view,
#   cascade_label, has_bacterial, has_viral, is_coinfection,
#   bacterial_labels, viral_labels, raw_labels
# ==============================================================================

SAVE_COLS = [
    'image_file', 'image_path', 'patient_id',
    'age', 'gender', 'view',
    'cascade_label', 'has_bacterial', 'has_viral', 'is_coinfection',
    'bacterial_labels', 'viral_labels', 'raw_labels',
]

for name, dset in [('train', df_train), ('val', df_val), ('test', df_test)]:
    save_df = dset[SAVE_COLS].copy()
    for col in ['bacterial_labels', 'viral_labels', 'raw_labels']:
        save_df[col] = save_df[col].apply(lambda x: '|'.join(x) if x else '')
    path = os.path.join(OUT_DIR, f'{name}.csv')
    save_df.to_csv(path, index=False)
    print(f'✅ Saved  {name}.csv  →  {len(save_df):,} rows  →  {path}')

# Save human-readable stats summary
stats_lines = [
    'NIH ChestX-ray14 — Filtered Dataset Statistics',
    '=' * 52,
    f'Generated from: {NIH_CSV}',
    '',
    'LABEL DEFINITIONS:',
    f'  BACTERIAL labels : {BACTERIAL_LABELS}',
    f'  VIRAL labels     : {VIRAL_LABELS}',
    f'  EXCLUDED labels  : {EXCLUDED_LABELS}',
    '',
    'SPLIT SIZES:',
]
for name, dset in [('Train', df_train), ('Val', df_val), ('Test', df_test)]:
    stats_lines.append(f'  {name}: {len(dset):,} images, {dset.patient_id.nunique():,} patients')
    for lbl in ['HEALTHY', 'BACTERIAL', 'VIRAL', 'CO-INFECTION']:
        cnt = int((dset['cascade_label'] == lbl).sum())
        stats_lines.append(f'    {lbl}: {cnt:,}')

with open(os.path.join(OUT_DIR, 'dataset_stats.txt'), 'w') as f:
    f.write('\n'.join(stats_lines))
print(f'\n✅ Saved  dataset_stats.txt')
print(f'\n📂 All files saved to: {OUT_DIR}')


# ==============================================================================
# CELL 11: SAMPLE IMAGE VIEWER — SEE WHAT EACH CLASS LOOKS LIKE
# ==============================================================================

def show_samples(dset, cascade_label_filter, n=4, title=''):
    subset_pool = dset[dset['cascade_label'] == cascade_label_filter]
    subset = subset_pool.sample(min(n, len(subset_pool)), random_state=42)
    fig, axes = plt.subplots(1, len(subset), figsize=(4 * len(subset), 4))
    if len(subset) == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=12, fontweight='bold')
    for ax, (_, row) in zip(axes, subset.iterrows()):
        try:
            img = Image.open(row['image_path']).convert('L')
            ax.imshow(img, cmap='gray')
        except Exception:
            ax.text(0.5, 0.5, 'Image\nnot found', ha='center', va='center',
                    transform=ax.transAxes)
        b_lbl = row['bacterial_labels'] if isinstance(row['bacterial_labels'], list) \
                else row['bacterial_labels'].split('|')
        v_lbl = row['viral_labels'] if isinstance(row['viral_labels'], list) \
                else row['viral_labels'].split('|')
        ax.set_title(
            f"{row['image_file']}\nB: {b_lbl}\nV: {v_lbl}",
            fontsize=7)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f'samples_{cascade_label_filter}.png'),
                dpi=100, bbox_inches='tight')
    plt.show()

for lbl in ['HEALTHY', 'BACTERIAL', 'VIRAL', 'CO-INFECTION']:
    cnt = int((df_train['cascade_label'] == lbl).sum())
    if cnt > 0:
        show_samples(df_train, lbl, n=4, title=f'{lbl}  (n={cnt:,} in train)')
    else:
        print(f'⚠️  No {lbl} images in train set')


# ==============================================================================
# CELL 12: FINAL SUMMARY
# ==============================================================================

print('╔══════════════════════════════════════════════════════════════╗')
print('║   NIH DATASET PREPARATION — COMPLETE                        ║')
print('╠══════════════════════════════════════════════════════════════╣')
print(f'║  Source      : NIH ChestX-ray14  (112,120 raw images)       ║')
print(f'║  After filter: {len(df):>6,} images kept                          ║')
print(f'║  Dropped     : {len(df_raw)-len(df):>6,} excluded-only images               ║')
print( '╠══════════════════════════════════════════════════════════════╣')
print( '║  PATHOGEN MAPPING:                                          ║')
print( '║  BACTERIAL → Consolidation, Pneumonia,                      ║')
print( '║               Atelectasis, Infiltration                     ║')
print( '║  VIRAL     → Infiltration, Pneumonia, Effusion, Edema       ║')
print( '║  EXCLUDED  → Emphysema, Fibrosis, Cardiomegaly,             ║')
print( '║               Nodule, Mass, Hernia, Pleural_Thickening,     ║')
print( '║               Pneumothorax                                  ║')
print( '╠══════════════════════════════════════════════════════════════╣')
for split_name, dset in [('TRAIN', df_train), ('VAL  ', df_val), ('TEST ', df_test)]:
    h  = int((dset.cascade_label == 'HEALTHY').sum())
    b  = int((dset.cascade_label == 'BACTERIAL').sum())
    v  = int((dset.cascade_label == 'VIRAL').sum())
    ci = int((dset.cascade_label == 'CO-INFECTION').sum())
    print(f'║  {split_name} : {len(dset):>6,} | H:{h:>5,} B:{b:>5,} V:{v:>5,} CI:{ci:>4,}    ║')
print( '╠══════════════════════════════════════════════════════════════╣')
print(f'║  Output files saved to: {OUT_DIR:<37}║')
print(f'║    train.csv  val.csv  test.csv  dataset_stats.txt          ║')
print(f'║    dataset_overview.png  samples_*.png                      ║')
print( '╠══════════════════════════════════════════════════════════════╣')
print( '║  NEXT STEP: Load these CSVs in your cascade pipeline        ║')
print( '║    df_train = pd.read_csv("nih_prepared/train.csv")         ║')
print( '║    df_val   = pd.read_csv("nih_prepared/val.csv")           ║')
print( '║    df_test  = pd.read_csv("nih_prepared/test.csv")          ║')
print( '╚══════════════════════════════════════════════════════════════╝')
