# Face Anti-Spoofing: A Comparative Deep Learning Study Using DenseNet-161 and CDCN++

This project explores two advanced deep learning architectures — DenseNet-161 and CDCN++ — in the context of face anti-spoofing, with a goal of enhancing the robustness and reliability of facial recognition systems. Built from the ground up, this work was independently designed, implemented, and evaluated by me as a focused research and engineering endeavor in biometric security.

Facial recognition systems are increasingly vulnerable to spoofing attacks, which exploit the limitations of visual input — including photos, videos, masks, and prosthetic alterations — to deceive systems into granting unauthorized access. To address this, my project investigates the effectiveness of two convolutional models: one based on pixel-wise supervision using DenseNet-161, and the other leveraging the central difference features of CDCN++.

My approach begins by training both models on the LCC-FASD dataset, a widely respected benchmark in face anti-spoofing, which contains thousands of real and spoofed facial images. All images are preprocessed through resizing, normalization, and stratified labeling to maintain balanced learning.

### DenseNet-161 with Pixel-Wise Supervision

The DenseNet-161 model is adapted here to operate with pixel-wise loss, enabling it to make fine-grained decisions across local regions of the face. This is critical in detecting texture differences and minute cues that typically escape coarser models. DenseNet's densely connected architecture allows for efficient gradient propagation and rich feature reuse — both of which improve the model’s ability to generalize across spoofing types.

Pixel-wise supervision means each image is not judged holistically, but is instead parsed down to individual pixel responses, ensuring detailed spatial reasoning. This supervision is implemented with a 1×1 convolutional layer followed by a linear classifier, allowing the system to produce a comprehensive map that clearly distinguishes genuine and spoofed regions.

### CDCN++ with Depth-Aware Feature Extraction

In contrast to the feature reuse strategy of DenseNet, CDCN++ focuses on learning depth-aware co-occurrence features. It replaces standard convolutions with central difference convolution operations — an innovation that highlights differences in local pixel intensity, crucial for texture- and depth-based spoof detection.

CDCN++ also integrates spatial attention layers that enable the model to focus on regions of the image that are most likely to contain spoofing cues. Though less granular than pixel-wise analysis, CDCN++ performs well on lower-resolution and noisy input, making it useful in real-world surveillance scenarios.

### Evaluation & Results

To evaluate performance, I trained both models across multiple epochs, tuning hyperparameters and tracking loss and accuracy on validation datasets. The final metrics are compelling:

- The DenseNet-161 model reached a classification accuracy of **90.4%**, with precision and recall values around **0.90**, and an F1-score that indicates strong balance between detecting genuine users and flagging spoof attempts.
- CDCN++, while innovative in its convolutional approach, plateaued at **76.8% accuracy**, with lower precision on spoof detection and slightly longer convergence times.
- Confusion matrices and classification reports highlight that DenseNet-161 consistently produced fewer false positives and false negatives across varied lighting and spoofing conditions.

Visualizations were included for both training and testing phases, showing steady convergence in DenseNet's learning curves and slightly noisier outcomes for CDCN++.

### Deployment

For user testing, I deployed the DenseNet-161 model using **Gradio**, a Python-based UI toolkit. The Gradio interface allows real-time image uploads and gives immediate feedback on whether the image is genuine or spoofed. Two sample screenshots — one of a real detection and one of a spoof detection — are provided in the repository to demonstrate performance.

This interface can be extended to camera-based input, making it suitable for access control use cases or mobile verification workflows. The backend is lightweight and can be integrated into larger security pipelines as a preliminary filter or validation module.

### Conclusion

While both models have merits, my findings suggest that **pixel-wise supervision using DenseNet-161** offers a more precise, scalable, and secure solution for face anti-spoofing. It generalizes well across spoofing types, converges quickly, and is easier to deploy in real-time applications.

Future extensions could explore:
- Combining RGB and depth data for multi-modal spoof detection
- Integrating adversarial training to defend against adaptive attacks
- Converting models to ONNX for use on mobile and edge devices
- Expanding dataset coverage with custom spoof scenarios

All training notebooks, evaluation results, and the full technical report are available in this repository.

This project was built entirely by me as an academic and engineering milestone in deep learning and computer vision security.

##  System Architecture & Model Flow

### Overall System Architecture
<img width="410" alt="Overall Architecture" src="https://github.com/user-attachments/assets/db99b9d1-edf9-4727-bc5b-4fb430f3a3b6" />

**Fig. 1:** High-level system architecture illustrating components involved in CNN-based face spoof detection.

---

### DeePixBiS Flow Diagram
<img width="358" alt="DeePixBiS Flow" src="https://github.com/user-attachments/assets/c1c130f9-6947-4d6b-8a7b-efef17ede17e" />

**Fig. 2:** DeePixBiS pipeline structure featuring pixel-wise supervision and attention blocks.

---

### CDCN++ Flow Diagram
<img width="402" alt="CDCN Flow" src="https://github.com/user-attachments/assets/8116cb27-dfa3-4fa3-8d45-521a69b70c0e" />

**Fig. 3:** CDCN++ flow diagram showing central difference convolution layers used to detect fine spoofing cues.

---

##  Output Comparisons: Genuine vs Spoof

### DenseNet-161 (Pixel-wise Supervision)

<img width="258" alt="Genuine DenseNet" src="https://github.com/user-attachments/assets/7c32e3e8-cf02-4cdb-8b44-970a12769014" />

**Fig. 4:** Output result from DenseNet-161 model for a **genuine face** image.

<img width="258" alt="Spoofed DenseNet" src="https://github.com/user-attachments/assets/5693dc6e-0410-48f5-a318-a07932a4113f" />

**Fig. 5:** Output result from DenseNet-161 model for a **spoofed** image.

---

### CDCN++ Model

<img width="240" alt="Genuine CDCN" src="https://github.com/user-attachments/assets/bb2f1a37-96db-4472-9aea-90fbbba17cef" />

**Fig. 6:** Output from CDCN++ model when tested with a **genuine** face image.

<img width="234" alt="Spoofed CDCN" src="https://github.com/user-attachments/assets/96365144-280f-4851-bb56-72da8e6c036c" />

**Fig. 7:** Output from CDCN++ model for a **spoofed** image.

---

##  Evaluation Metrics

<img width="321" alt="Confusion Matrix DenseNet" src="https://github.com/user-attachments/assets/1f0b3182-573a-406a-bbbd-3dd26c527c8f" />

**Fig. 8:** Confusion matrix for DenseNet-161 model showing strong performance with fewer false positives.

<img width="305" alt="Confusion Matrix CDCN" src="https://github.com/user-attachments/assets/90d07d87-6080-496a-83c5-ed24eec8d6a2" />

**Fig. 9:** Confusion matrix for CDCN++ model with slightly more classification ambiguity.

<img width="402" alt="Classification DenseNet" src="https://github.com/user-attachments/assets/2666829a-4050-4387-9e48-e5661991a80d" />

**Fig. 10:** Classification report showing precision, recall, and F1-score of DenseNet-161.

<img width="391" alt="Classification CDCN" src="https://github.com/user-attachments/assets/74252665-1824-4cf0-abd7-8d2a33a5ce80" />

**Fig. 11:** Classification report for CDCN++ model on the same dataset.

---

##  Gradio Interface Output

<img width="381" alt="Gradio Real" src="https://github.com/user-attachments/assets/352a3300-d72b-44ed-bbb2-68abd26b762e" />

**Fig. 12:** Gradio web interface correctly identifying a **real face** image as genuine.

<img width="372" alt="Gradio Spoof" src="https://github.com/user-attachments/assets/3af4d2d2-6629-4180-a092-08e3e3a839c6" />

**Fig. 13:** Gradio output showing a **spoofed** image being accurately flagged as fake.

---

##  Model Performance Comparison

<img width="402" alt="Accuracy Comparison" src="https://github.com/user-attachments/assets/24a3cc8a-ae56-4ea3-8e1b-9592e642cef6" />

**Fig. 14:** Accuracy comparison chart: DenseNet-161 (90%) vs CDCN++ (77%) on test dataset.

---

