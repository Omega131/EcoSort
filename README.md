<h1 align="center">EcoSort</h1>

<h2 align="center"><strong>Efficient and accurate waste classification for edge devices.</strong><br> Sort smart. Save the planet.</h2>

<details>
<summary style="font-size: 24px; font-weight: bold; cursor: pointer;"><h1>Table of Contents</h1></summary>

- [Architecture Diagram](#architecture-diagram)
- [The Problem](#the-problem)
- [Our solution](#our-solution)
- [Model Constraints & Optimization](#model-constraints--optimization)
- [Training Methodology](#training-methodology)
- [Deployment](#deployment)
- [License](#license)

</details>

# Architecture Diagram

*(Placeholder for EcoSort Architecture Diagram)*

## The Problem

Waste sorting and recycling is crucial for ecological sustainability, but manual sorting is inefficient and error-prone. While deep learning models can automate this process, deploying them on edge devices (like smart bins or mobile applications) poses significant challenges:

- **High Resource Requirements:** Traditional computer vision models are too large and slow for edge hardware.
- **Accuracy Trade-offs:** Shrinking models to fit on smaller devices often drastically reduces their ability to classify complex waste types accurately.
- **Strict Deployment Constraints:** Edge hardware typically imposes strict limits on model file size, memory footprint, and CPU latency.

## Our solution

EcoSort tackles these challenges by utilizing a highly optimized, compact convolutional neural network architecture specifically tailored for edge deployment. Our model guarantees high accuracy while strictly adhering to hardware constraints:

1. **Lightweight Convolutional Blocks**  
   We utilize depthwise separable convolutions to extract rich spatial features while keeping the parameter count incredibly low.

2. **Inverted Bottleneck Design**  
   By expanding the channel dimension before applying spatial filters and projecting back down, we maximize representational power without a huge performance penalty.

3. **Knowledge Distillation Integration**  
   The architecture is purposefully designed to act as an eager student network, perfectly capable of capturing complex class boundaries learned by a massive teacher model.

## Model Constraints & Optimization

Our final `model.onnx` successfully passes rigorous verification checks and meets all edge-deployment limits:

- **Total Parameters:** < 350,000 Limit
- **File Size:** < 1.50 MB Limit
- **CPU Latency:** Extremely fast inference on edge hardware

## Training Methodology

EcoSort leverages **Knowledge Distillation** to train the compact student model efficiently.
- **Teacher Model:** A larger, highly accurate teacher model (`teacher.pth`) guides the learning process.
- **Student Model:** Our lightweight architecture acts as the student, learning to mimic the teacher's soft targets.
- **Optimization:** We train the student to perfectly match the teacher's performance, retaining accuracy with just a fraction of the computational cost.

## Deployment

To deploy the model or train it locally:

```bash
# 1. Install dependencies
pip install -r starter_kit/requirements.txt

# 2. Train the student model (Knowledge Distillation)
python starter_kit/train_student.py --data_dir . --teacher_weights teacher.pth

# 3. Verify the exported ONNX model against constraints
python starter_kit/verify_submission.py
```

## License

[MIT License](LICENSE)