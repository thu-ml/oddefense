<div align="center">
  <h3>On the Importance of Backbone to the Adversarial Robustness of Object Detectors</h3>
  <a href="https://arxiv.org/pdf/2305.17438">
        <img alt="Build" src="https://img.shields.io/badge/arXiv%20paper-2305.17438-b31b1b.svg">
  </a>
</div>


<h2 id="quick-start">Quick Start</h2>

This is the official implementation for ''On the Importance of Backbone to the Adversarial Robustness of Object Detectors'', IEEE TIFS 2025.

<h3>Preparation</h3>

  ```sh
  conda create -n oddefense python=3.10
  conda activate oddefense

  conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia

  pip install -U openmim
  mim install mmcv-full==1.7.0
  pip install mmdet==2.28.0
  pip install -r requirements.txt
  ```

  Download pretrained ResNet-50 backbone: <a href='https://huggingface.co/suixin1424/oddefense/blob/main/resnet50_linf_eps4_pure.pth'>resnet-50 pretrained </a>
  Download pretrained ConvNeXt-T backbone: <a href='https://huggingface.co/suixin1424/oddefense/blob/main/convnext_tiny_mmcls-linf-eps-4-advan.pth'>convnext-t pretrained </a>
  

<h3>Train and Evaluate</h3>

1. **Modify Config Files**  
   Update the following variables in the config files (e.g., `frcnn/faster_rcnn_r50_fpn_1x_coco_freeat_all.py`):
   - `checkpoint_at`
   - `data_root`
   - `work_dir`

2. **Training**  
   Run the following command to start training:
    ```bash
    bash tools/dist_train.sh [config_file] [num_gpus]
    ```

3. **Evaluation**  
  Run the following command to evaluate your model:
    ```bash
    bash tools/dist_test.sh [config_file] [ckpt_path] [num_gpus] --eval bbox
    ```

<h2 id="models">Models</h2>

| **Model**       | **Config File**                                                                                     | **Checkpoint**                          |
|------------------|-----------------------------------------------------------------------------------------------------------|------------------------------------------|
| Faster-RCNN  | [`faster_rcnn_r50_fpn_1x_coco_freeat_all.py`](frcnn/faster_rcnn_r50_fpn_1x_coco_freeat_all.py)            | <a href='https://huggingface.co/suixin1424/oddefense/blob/main/frcnn_at.pth'> click to download </a> |
| FCOS            | [`fcos_r50_caffe_fpn_gn-head_1x_coco_freeat_all.py`](fcos/fcos_r50_caffe_fpn_gn-head_1x_coco_freeat_all.py)                                       | <a href='https://huggingface.co/suixin1424/oddefense/blob/main/fcos_at.pth'> click to download </a>            |
| DN-DETR         | [`dn_detr_r50_8x2_12e_coco_freeat_all.py`](dn_detr/dn_detr_r50_8x2_12e_coco_freeat_all.py)                                   | <a href='https://huggingface.co/suixin1424/oddefense/blob/main/dndetr_at.pth'> click to download </a>         |
| Faster-RCNN ConvNeXt   | [`faster_rcnn_convnext_fpn_1x_coco_freeat_all.py`](frcnn/faster_rcnn_convnext_fpn_1x_coco_freeat_all.py)                                   | <a href='https://huggingface.co/suixin1424/oddefense/blob/main/frcnn_convnext.pth'> click to download </a>         |
| FCOS ConvNeXt     | [`fcos_convnext_caffe_fpn_gn-head_1x_coco_freeat_all.py`](fcos/fcos_convnext_caffe_fpn_gn-head_1x_coco_freeat_all.py)                                       | <a href='https://huggingface.co/suixin1424/oddefense/blob/main/fcos_convnext.pth'> click to download </a>            |
| DN-DETR ConvNeXt   | [`dn_detr_convnext_8x2_12e_coco_freeat_all.py`](dn_detr/dn_detr_convnext_8x2_12e_coco_freeat_all.py)                                   | <a href='https://huggingface.co/suixin1424/oddefense/blob/main/dndetr_convnext.pth'> click to download </a>         |

<h3>
Acknowledgement
</h3>

If you find that our work is helpful to you, please star this project and consider cite:

```
@article{li2025importance,
  title={On the Privacy Effect of Data Enhancement via the Lens of Memorization},
  author={Li, Xiao and Chen, Hang and Hu, Xiaolin},
  journal={IEEE Transactions on Information Forensics and Security}, 
  year={2025}
  }
```