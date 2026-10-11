<h1 align="center">EcoSort</h1>

<img width="2400" height="1792" alt="Gemini_Generated_Image_770q63770q63770q" src="https://github.com/user-attachments/assets/04dbce26-55da-42f9-8bff-537ebd8dd04d" />

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

<img width="979" height="550" alt="ecosort architecture (1)" src="https://github.com/user-attachments/assets/fb789fe7-3410-4454-9cb1-93aebd048096" />

<img width="1504" height="323" alt="ecosort workflow (1)" src="https://github.com/user-attachments/assets/e94a079b-9c82-458b-a484-3e7e3933dc57" />

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

## Deployment & Live Web Application

### 1. Launch the Interactive Web App & REST API
Deploy and test the model immediately on your own images or live camera:

```bash
# Install dependencies
pip install -r starter_kit/requirements.txt

# Start the web app and API server
python run_app.py
```

- **Interactive Web Interface:** Open [http://localhost:8000](http://localhost:8000) in your browser.
  - 📁 **Drag-and-Drop File Upload:** Upload custom photos of waste items.
  - 📹 **Live Camera Scanner:** Real-time camera viewfinder with instant frame capture.
  - ⚡ **1-Click Sample Test Drive:** Instant classification across 6 waste categories.
  - 📊 **Confidence Meters & Recycling Guidance:** Actionable bin recommendations (Blue, Yellow, Green, Gray), preparation steps, and environmental impact facts.
- **Interactive Swagger REST API Docs:** Available at [http://localhost:8000/docs](http://localhost:8000/docs)

### 2. Edge / IoT API Usage (`curl` & Python)
Integrate EcoSort directly into smart bins or microcontrollers:

```bash
# Classify an image via curl
curl -X POST "http://localhost:8000/api/predict" \
     -F "file=@your_waste_image.jpg"
```

```python
import requests

with open("item.jpg", "rb") as f:
    response = requests.post("http://localhost:8000/api/predict", files={"file": f})
result = response.json()
print(f"Detected: {result['predicted_class']} ({result['confidence']}%) in {result['latency_ms']} ms")
print(f"Recommended Bin: {result['guidance']['bin_name']}")
```

### 3. Verification & Local Training
```bash
# Verify the exported ONNX model against competition constraints
python starter_kit/verify_submission.py

# Optional: Retrain the student model via Knowledge Distillation
python starter_kit/train_student.py --data_dir . --teacher_weights teacher.pth
```

## License

[MIT License](LICENSE)
