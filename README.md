<div align="center">

# ATLAS Navigator

### Active Task-driven LAnguage-embedded Gaussian Splatting

[Dexter Ong](https://dexterong.com/) · [Yuezhan Tao](https://tyuezhan.github.io/) · [Varun Murali](https://varunmurali1.github.io/) · [Igor Spasojevic](https://scholar.google.com/citations?user=IyoEsBQAAAAJ&hl=en) · [Vijay Kumar](https://www.kumarrobotics.org/) · [Pratik Chaudhari](https://pratikac.github.io/)

**IEEE Transactions on Field Robotics (T-FR), 2026**

[![Project Page](https://img.shields.io/badge/Project-Page-2563eb)](https://ongdexter.github.io/atlasnav/)
[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b)](https://arxiv.org/abs/2502.20386)
[![Video](https://img.shields.io/badge/Video-YouTube-ff0000)](https://youtu.be/vjuLE1k0htA)

</div>

ROS 1 implementation of [ATLAS Navigator](https://ongdexter.github.io/atlasnav/) for task-driven navigation using language-embedded Gaussian splatting.

## Build

Requires Linux, an NVIDIA GPU, Docker, and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). The image includes ROS Noetic, CUDA, PyTorch, the rasterizers, `mpl_py`, the Jackal tracker, and CLIP-DINOiser.

Clone the repository and its pinned dependencies:

```bash
git clone --recurse-submodules https://github.com/grasp-lyrl/atlasnav.git
cd atlasnav
```

Docker copies these sources from your checkout.

Before building, configure:

- Camera dimensions, intrinsics, and camera-to-body extrinsics (`cam_config.cam2body`) in [active_gs_robot.py](active_3dgs/config/active_gs_robot.py). The defaults expect registered depth in **millimeters** (`png_depth_scale=1000.0`) and `geometry_msgs/PoseStamped` poses. Intrinsics come from this file, not `CameraInfo`. `cam2body` is a 4×4 matrix mapping optical camera coordinates into the robot body frame, with translation in meters.
- Robot frames, planner/tracker settings, and `use_language_features` (currently `true`) in [local_planner.yaml](active_3dgs/config/local_planner.yaml).

From the repository directory:

```bash
docker build -t active-3dgs:ros1 -f docker/Dockerfile .
```

## Deployment

Start your camera driver, pose source, and TF tree outside Docker. Start the container:

```bash
docker run --rm --name active-3dgs --gpus all --network host --shm-size=1g \
  -v active-3dgs-model-cache:/root/.cache/huggingface \
  -v active-3dgs-output:/root/.ros/active_3dgs active-3dgs:ros1
```

CLIP backbone weights download on first use and persist in the `active-3dgs-model-cache` volume.

Start your tracker/controller separately. For the included Jackal tracker, configure its pose input and `/twist_auto` output, then run in another terminal:

```bash
docker run --rm --gpus all --network host active-3dgs:ros1 \
  roslaunch jackal_mp_tracker jackal_mp_tracker_robot.launch
```

Change the language target or save a map from another terminal:

```bash
docker exec active-3dgs active-3dgs-entrypoint \
  rosservice call /set_task "task: 'tree,background'"

docker exec active-3dgs active-3dgs-entrypoint \
  rosservice call /save_gs_map "{}"
```

Output persists in the `active-3dgs-output` volume at `/root/.ros/active_3dgs` inside the container. The `/set_task` service requires language features to be enabled.

## Acknowledgements

We thank the authors and contributors of the following open-source projects:

- [SplaTAM](https://github.com/spla-tam/SplaTAM)
- [MPL](https://github.com/sikang/motion_primitive_library)
- [Differentiable Gaussian Rasterization with Depth](https://github.com/JonathonLuiten/diff-gaussian-rasterization-w-depth)

## Citation

If you find our work useful, please cite:

```bibtex
@misc{ong2026atlas,
  title={ATLAS Navigator: Active Task-driven LAnguage-embedded Gaussian Splatting},
  author={Dexter Ong and Yuezhan Tao and Varun Murali and Igor Spasojevic and Vijay Kumar and Pratik Chaudhari},
  year={2025},
  eprint={2502.20386},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2502.20386},
}
```
