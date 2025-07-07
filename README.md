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

