# IMPLEMENT SIMPLE JPEG2000 IMAGE COMPRESSION ALGORITHM

This project implements a simplified version of the JPEG2000 image compression standard using Python. JPEG2000 is a wavelet-based image compression standard that offers superior compression efficiency and image quality compared to traditional JPEG, especially at higher compression ratios. 

The implementation focuses on the core components of JPEG2000:
- Discrete Wavelet Transform (DWT) for image decomposition
- Coefficient quantization
- Entropy coding for data compression
- Proper handling of color (RGB) images

1. Discrete Wavelet Transform (DWT) – Image Decomposition
   Each color channel (Red, Green, Blue) of the image is processed separately. The image is decomposed using the Discrete Wavelet Transform (DWT), which breaks it into multiple sub-bands of different frequency components. The wavelet used is configurable (e.g., db4, bior4.4), and the decomposition is done over several levels (e.g., 4 or 5).

   DWT enables multi-resolution representation of the image. It separates the image into:
   - Approximation coefficients (low-frequency) – contain most of the image’s important visual information.
   - Detail coefficients (high-frequency) – capture edges, textures, and fine details.
2. Quantization
   After DWT, the resulting coefficients (floating-point values) are quantized—converted to lower-precision integers by dividing them by a quantization_step and rounding the result. Quantization reduces the precision of coefficients, allowing for much higher compression. This step introduces loss, but most visual quality is retained if approximation coefficients are preserved well. Quantization is one of the main contributors to compression efficiency.
3.  Entropy Coding – Data Compression
   Entropy coding exploits patterns and redundancies in data. Delta encoding often produces smaller numbers (more compressible), and Zlib efficiently compresses repeating patterns. This step significantly reduces file size without losing any additional information beyond quantization.
4. Color Handling – RGB Channel Processing
   The image is split into three separate channels: Red, Green, and Blue. Each channel is compressed independently using the same DWT, quantization, and entropy coding pipeline. During decompression, the channels are reconstructed and then combined back into an RGB image. Processing each color channel individually allows better control over compression and avoids introducing color distortions. This approach mimics how JPEG and JPEG2000 treat color in real-world applications.

Quality Evaluation:
- PSNR (Peak Signal-to-Noise Ratio): Measures how much the reconstructed image differs from the original. Higher PSNR generally means better quality.
- SSIM (Structural Similarity Index): Focuses on perceived image quality by comparing structure, contrast, and luminance. SSIM is often a better indicator of visual quality than PSNR.

