# Third-party components

This implementation uses public pretrained components through their upstream libraries. Install and use each component under its own license and model terms. No third-party model weights are redistributed here.

- PyTorch and Torchvision: tensor operations and AlexNet weights used by LPIPS.
- Hugging Face Diffusers, Transformers, Accelerate and PEFT: diffusion pipelines, VAE/CLIP encoders and LoRA loading.
- LPIPS: learned perceptual image distance with the AlexNet backbone.
- TorchMetrics and scikit-image: SSIM and HOG measurements.
- Ultralytics: YOLO instance segmentation, including the segmentation model's applicable license.
- NumPy and Pillow: numerical computation and image preprocessing.

Model directories must include their original configuration and license/model-card files.
