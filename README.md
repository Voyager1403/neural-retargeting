# Neural Retargeting

Code for the paper "Human-Robot Motion Retargeting via Neural Latent Optimization" submitted to IROS2021

## Prerequisite

- [**PyTorch**](https://pytorch.org/get-started/locally/) Tensors and Dynamic neural networks in Python with strong GPU acceleration
- [**pytorch_geometric**](https://github.com/rusty1s/pytorch_geometric) Geometric Deep Learning Extension Library for PyTorch

## Get started

**Training**
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --cfg './configs/train/yumi.yaml'
```

**Inference**
```bash
CUDA_VISIBLE_DEVICES=0 python inference.py --cfg './configs/inference/yumi.yaml'
```
## Citation

If you find this project useful in your research, please cite this paper.

```
@article{zhang2021human,
  title={Human-Robot Motion Retargeting via Neural Latent Optimization},
  author={Zhang, Haodong and Li, Weijie and Liang, Yuwei and Chen, Zexi and Cui, Yuxiang and Wang, Yue and Xiong, Rong},
  journal={arXiv preprint arXiv:2103.08882},
  year={2021}
}
```
