<div style="float: right; margin-left: 20px;">
  <img alt="BVR Chirp Logo" src="logo.png" width="240" />
</div>

# bvr_web_pup



BVR Web Pup is an endpoint server that performs object detection on images on incoming images and returns back the results.

### THIS PROJECT IS IN ALPHA. It should work, but don't expect too much yet.

# Building

You need to have [bvr_detect](https://github.com/CeeBeeEh/bvr_detect) cloned and available for this project.

Assuming they are placed next to each other like this:

```
..
bvr_detect\
bvr_web_pup\
```

Your Cargo.toml should have:

`bvr_detect = { version = "0.2.0", path = "../bvr_detect" }`

# Installation

Currently, there's no installation process. You would just run the binary with your parameters.

# Building

`cargo build --release`

# Running

`./BvrWebPup <options>`

Options you can set:

| Description                                                                   | Flag                   | Input                            |
|-------------------------------------------------------------------------------|------------------------|----------------------------------|
| Path to weights/model file                                                    | `-m --model-path`      | File path                        |
| Path to labels file                                                           | `-c --classes-path`    | File path                        |
| Path to OnnxRuntime library                                                   | `-l --lib-path-ort`    | Path to .so library file         |
| Device to use for inference                                                   | `-d --device`          | CPU, CUDA, TensorRT              |
| Detection processing method to use                                            | `-r --processing-type` | ORT, Torch, Python               |
| Specify YOLO Version                                                          | `-y --yolo-version`    | v4, v5, v6, v7, v8, v9, v10, v11 |
| Split extra wide images (dual lens cameras) to process each half individually | `-s `                  | \<bool flag>                     |
| Set detection threshold (default is 0.4)                                      | `-t --threshold`       | 0.1 - 1.0 (higher is stricter)   |
| Input width for the model                                                     | `-w --width`           | A value divisible by 32          |
| Input height for the model                                                    | `-e --height`          | A value divisible by 32          |
| Set a manual port for the web server (default 3000)                           | `-p --port`            | Any available port on the system |
| Specify logging level (default INFO)                                          | `--log-level`          | ERROR, WARN, INFO, DEBUG, TRACE  |

Note: The case for the inputs is not important. Everything is converted to lowercase when evaluating.

# TODO:
- [x] Get this code published
- [ ] Ability to set thresholds for each camera, and each object per camera
- [ ] Web interface to see detection results and modify settings 
  - [ ] Web interface
  - [ ] Option to save/view previous detections
    - [ ] Ability to easily re-process images with different settings for testing
  - [ ] Page to test detections with uploaded image
