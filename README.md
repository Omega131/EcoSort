<h1 align="center">EcoSort</h1>

<h2 align="center"><strong>Efficient and accurate waste classification using an optimized MobileNetV4 architecture.</strong><br> Sort smart. Save the planet.</h2>

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

EcoSort tackles these challenges by utilizing a highly optimized, custom **MobileNetV4** architecture specifically tailored for edge deployment. Our model guarantees high accuracy while strictly adhering to hardware constraints:

1. **Universal Inverted Bottleneck (UIB)**  
   We integrate MobileNetV4's UIB blocks with configurable paths (Standard, Extra Depthwise, and ConvNeXt-style) for optimal spatial and channel mixing without bloating the parameter count.

2. **MobileMQA (Multi-Query Attention)**  
   By pooling Keys and Values, our custom MobileMQA module brings global reasoning to the network with an $O(N)$ complexity, adding negligible parameters while significantly boosting feature extraction capabilities.

3. **Delayed Spatial Downsampling**  
   We re-configured the block architectures to delay stride-2 downsampling until later stages. This preserves high-resolution feature maps longer, greatly improving classification accuracy on small or complex waste items.

4. **Maximized Constraint Utilization**  
   Instead of just making the model as small as possible, we dynamically scale the `expand_ratio` and channel widths in the deeper layers to maximize our parameter budget (~335k out of the 350k limit) and file size limit (1.37MB out of the 1.50MB limit), ensuring peak performance within our strict constraints.

## Model Constraints & Optimization

Our final `model.onnx` successfully passes rigorous verification checks and meets all edge-deployment limits:

- **Total Parameters:** ~335,030 (Limit: 350,000)
- **File Size:** 1.37 MB (Limit: 1.50 MB)
- **CPU Latency:** ~2.03 ms/image (Extremely fast inference)
- **Validation Accuracy:** 100.00%

## Training Methodology

EcoSort leverages **Knowledge Distillation** to train the compact student model efficiently.
- **Teacher Model:** A larger, highly accurate teacher model (`teacher.pth`) guides the learning process.
- **Student Model:** Our custom MobileNetV4 architecture acts as the student, learning to mimic the teacher's soft targets.
- **Optimization:** Trained over 50 epochs using optimized hyper-parameters (Temperature = 3.0, Alpha = 0.5), the student achieves 100% accuracy, matching the teacher's performance with a fraction of the computational cost.

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