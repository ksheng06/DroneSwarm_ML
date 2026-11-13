# DroneSwarm_ML

Machine-learning prototype for autonomous drone swarm disaster response.

## Project Focus

Early detection of forest fires and smoke-generating events in Indonesia that may contribute to regional haze affecting Malaysia.

## Implemented System

The prototype combines:

- MobileNetV2 transfer learning for Fire, Smoke and Neutral classification
- Qwen3-VL 4B as a secondary multimodal visual verifier
- Multi-model disagreement detection
- Five-level safety-oriented fusion policy
- Human-operator escalation
- Automatic incident logging
- Folder-based batch analysis

## MobileNetV2 Evaluation

Final unseen test set:

- 300 images
- Test accuracy: 91.67%
- Fire recall: 88%
- Smoke recall: 91%
- Neutral recall: 96%

## Fusion Policy

- Level 0: Normal
- Level 1: Model conflict / low confidence
- Level 2: Smoke hazard
- Level 3: Fire hazard
- Level 4: Fail-safe

## Main Programs

- `train_model.py` - trains MobileNetV2
- `evaluate_model.py` - evaluates the trained model
- `predict.py` - MobileNetV2 single-image inference
- `multimodel_agent.py` - MobileNetV2 + Qwen3-VL safety fusion
- `batch_analyze.py` - automatic multi-image batch analysis
- `prepare_dataset.py` - prepares validation data
- `check_images.py` - dataset compatibility validation
- `convert_neutral.py` - converts incompatible test images

## Requirements

- Python 3.13
- TensorFlow
- Ollama
- Qwen3-VL 4B

Qwen3-VL must be available locally through Ollama:

`ollama pull qwen3-vl:4b`

## Scope

This repository implements a software prototype. Physical UAV hardware, flight control, sensor hardware and deployed swarm networking are outside the implemented scope.