_base_ = [
    '../configs/_base_/models/mask_rcnn_r50_fpn.py',
    '../configs/_base_/datasets/coco_panoptic.py',
    '../configs/_base_/schedules/schedule_1x.py', '../configs/_base_/default_runtime.py'
]

dataset_type = 'CocoPanopticDataset'
data_root = '/home/share/datasets/coco/'
data_root_aux = '/home/lixiao/data3/coco_panoptic/panoptic/'
work_dir = "/home/lixiao/ssd/workdir/oddefense/frcnn/coco_panoptic_adv_benign"


# adversarial trainging and eval config
free_m = 4
times = 2


# full version

adv_cfg = dict(
    adv_flag=True,
    adv_type="com", # assert in ["all", "mtd", "cwa", "ours"]
    free_m=free_m,
    epsilon=4)


test_adv_cfg = dict(
    adv_flag=True,
    adv_type="cls", # assert in ["cls", "reg", "cwa", "dag", "ours"]
    step_size=1,
    epsilon=4,
    num_steps=10,
)

model = dict(
    type='PanopticFPN',
    backbone=dict(frozen_stages=1, init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    train_cfg=dict(rcnn=dict(clip=6)),
    semantic_head=dict(
        type='PanopticFPNHead',
        num_things_classes=80,
        num_stuff_classes=53,
        in_channels=256,
        inner_channels=128,
        start_level=0,
        end_level=4,
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        conv_cfg=None,
        loss_seg=dict(
            type='CrossEntropyLoss', ignore_index=255, loss_weight=0.5)),
    panoptic_fusion_head=dict(
        type='HeuristicFusionHead',
        num_things_classes=80,
        num_stuff_classes=53),
    test_cfg=dict(
        panoptic=dict(
            score_thr=0.6,
            max_per_img=100,
            mask_thr_binary=0.5,
            mask_overlap=0.5,
            nms=dict(type='nms', iou_threshold=0.5, class_agnostic=True),
            stuff_area_limit=4096)))

custom_hooks = []


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadPanopticAnnotations',
        with_bbox=True,
        with_mask=True,
        with_seg=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='SegRescale', scale_factor=1 / 4),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadPanopticAnnotations',
        with_bbox=True,
        with_mask=True,
        with_seg=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='SegRescale', scale_factor=1 / 4),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_semantic_seg'])
]

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=int(500/free_m),
    warmup_ratio=0.001,
    step=[times*10//free_m])

runner = dict(type='AdvEpochBasedRunner', max_epochs=times*12//free_m)

optimizer_config = dict(_delete_=True,
                    type='AdvOptimizerHook',
                    grad_clip=dict(max_norm=100, norm_type=2)) # ignore previous setting
optimizer = dict(
    _delete_ = True,
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.1,
    paramwise_cfg=dict(norm_decay_mult=0.,
        custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0)},
        bypass_duplicate=True
    )
)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root_aux + 'panoptic_train2017.json',
        img_prefix=data_root + 'train2017/',
        seg_prefix=data_root_aux + 'panoptic_train2017/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root_aux + 'panoptic_val2017.json',
        img_prefix=data_root + 'val2017/',
        seg_prefix=data_root_aux + 'panoptic_val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root_aux + 'panoptic_val2017.json',
        img_prefix=data_root + 'val2017/',
        seg_prefix=data_root_aux + 'panoptic_val2017/',
        pipeline=test_pipeline))

auto_scale_lr = dict(enable=False, base_batch_size=16)