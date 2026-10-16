# Reduced protocol shared by baseline A (PTv3) and baseline B (DITR) — the ONE place it is defined.
#
# STATUS: DRAFT (M1). Only values the user has fixed are filled in. Everything set to None is
# chosen in M3 from measured s/iter and memory, approved by the user, and then this file is
# frozen (status = "frozen", date + commit recorded in docs/decisions.md). After freezing,
# changes need explicit user approval.
#
# Pointcept-style config (plain Python), so experiment configs can inherit it via _base_.

protocol = dict(
    name="reduced_v1",
    status="draft",
    frozen_on=None,  # "YYYY-MM-DD @ <commit>" once frozen in M3
)

# ---- data (M2) -------------------------------------------------------------------------------
dataset = dict(
    name="scannetpp_v2",
    val_scenes="all",  # all 50 scenes of the official semantic val split
    train_scenes=200,  # fixed random subset of the train split
    train_subset_seed=0,
    scene_list="configs/scenes/reduced_v1.txt",  # written in M2, committed, never regenerated
    num_classes=100,  # ScanNet++ semantic benchmark classes
    ignore_index=-1,
    # anonymized regions are magenta (255, 0, 255) in meshes/images: color -> 0 + validity flag
    # channel; masked pixels in images (M2 implements, details in docs/decisions.md)
    anonymized_color=(255, 0, 255),
)

# ---- model / precision (T4: no FlashAttention 2, no bf16) -----------------------------------
grid_size = 0.02
enable_amp = True
amp_dtype = "float16"
enable_flash = False
upcast_attention = True  # proposed for fp16 stability in the non-flash path; confirm in M3
upcast_softmax = True
patch_size = None  # M3: reduced from 1024 to fit 16 GB
point_max = None  # M3: cap on points per crop (SphereCrop)

# ---- schedule (M3 proposes, user decides) ---------------------------------------------------
num_gpus = 2  # DDP on 2x T4
batch_size_per_gpu = None  # 1-2
epochs_or_iters = None  # sized so one run fits in one Kaggle session (<= 11 h incl. eval)
checkpoint_every_min = 30  # with automatic resume
seeds = (0, 1, 2)

# ---- evaluation (official ScanNet++ toolbox, M2) ---------------------------------------------
evaluation = dict(
    toolbox="third_party/scannetpp semantic/eval/eval_semantic.py",
    on="mesh_aligned_0.05.ply vertices",  # per-vertex predictions
    exclude_anonymized=True,
    metrics=("top1_miou", "top3_miou", "per_class_iou"),
)

# ---- DITR-specific (M5): follow DITR's indoor settings ----------------------------------------
ditr = dict(
    image_source="iphone",  # iPhone RGB + sensor depth first; DSLR + rendered depth is a later option
    frame_pool_per_scene=None,  # ~20-30 decoded frames per scene (M5)
    views_train=10,  # random subset per sample
    views_eval=10,  # temporally equidistant
    depth_occlusion_check=True,
    distance_filter_m=(1.0, 4.0),
    inject_levels="all_decoder",
    encoder="vit_large_patch14_reg4_dinov2",  # timm name used by DITR's FrozenDINOv2("large")
    image_size=(420, 560),  # DITR ScanNet setting -> 30x40 patches of 14 px
    cache_features="fp16 per-view patch maps + point->(view, patch) indices",
)
