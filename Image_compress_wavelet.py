import numpy as np
import pywt
from PIL import Image
import struct
import os
import zlib
import pickle
import numpy as np
import pywt
from PIL import Image
import struct
import os
import zlib
import pickle
from math import log10, sqrt
from skimage.metrics import structural_similarity as ssim
class JPEG2000:
    def __init__(self, wavelet='bior4.4', levels=5, quantization_step=16):
        self.wavelet = wavelet
        self.levels = levels
        self.quantization_step = quantization_step
    def compress(self, input_image_path, output_file_path):
        # Load image in RGB mode
        img = Image.open(input_image_path).convert('RGB')
        img_array = np.array(img, dtype=np.float32)
        # Process each channel separately
        compressed_channels = []
        for channel in range(3):  # R, G, B
            channel_data = img_array[:, :, channel]
            # Perform DWT
            coeffs = self._perform_dwt(channel_data)
            # Quantize coefficients
            quantized_coeffs = self._quantize(coeffs)
            # Entropy coding
            compressed_data = self._entropy_encode(quantized_coeffs)
            compressed_channels.append(compressed_data)
        # Combine all channels
        combined_data = self._combine_channels(compressed_channels)
        # Write to file with original shape (height, width, channels)
        self._write_to_file(combined_data, output_file_path, img_array.shape)
        return combined_data
    def decompress(self, input_file_path, output_image_path):
        # Read from file
        compressed_data, original_shape = self._read_from_file(input_file_path)
        # Split channels
        channel_data_list = self._split_channels(compressed_data, 3)  # Always 3 channels for RGB
        # Process each channel
        reconstructed_channels = []
        for channel_data in channel_data_list:
            # Entropy decoding
            quantized_coeffs = self._entropy_decode(channel_data)
            # Dequantize coefficients
            coeffs = self._dequantize(quantized_coeffs)
            # Perform inverse DWT
            reconstructed_channel = self._perform_inverse_dwt(coeffs, original_shape[:2])
            reconstructed_channels.append(reconstructed_channel)
        # Combine channels
        reconstructed_img = np.stack(reconstructed_channels, axis=-1)
        # Save image
        reconstructed_img = np.clip(reconstructed_img, 0, 255).astype(np.uint8)
        Image.fromarray(reconstructed_img).save(output_image_path)
        return reconstructed_img
    def _combine_channels(self, channel_data_list):
        # Combine multiple channel data with length prefixes
        combined = struct.pack('I', len(channel_data_list))
        for data in channel_data_list:
            combined += struct.pack('I', len(data)) + data
        return combined
    def _split_channels(self, combined_data, num_channels):
        # Split combined data back into channels
        offset = 4
        channel_data_list = []
        for _ in range(num_channels):
            data_len = struct.unpack('I', combined_data[offset:offset + 4])[0]
            offset += 4
            channel_data = combined_data[offset:offset + data_len]
            channel_data_list.append(channel_data)
            offset += data_len
        return channel_data_list
    def _perform_dwt(self, img_array):
        coeffs = pywt.wavedec2(img_array, self.wavelet, level=self.levels)
        return coeffs
    def _perform_inverse_dwt(self, coeffs, original_shape):
        reconstructed_img = pywt.waverec2(coeffs, self.wavelet)
        # Crop to original size (wavelet transform may pad the image)
        h, w = original_shape
        reconstructed_img = reconstructed_img[:h, :w]
        return reconstructed_img
    def _quantize(self, coeffs):
        quantized_coeffs = []
        for coeff in coeffs:
            if isinstance(coeff, tuple):
                # For detail coefficients (horizontal, vertical, diagonal)
                quantized_tuple = tuple(np.round(c / self.quantization_step) for c in coeff)
                quantized_coeffs.append(quantized_tuple)
            else:
                # For approximation coefficients
                quantized_coeffs.append(np.round(coeff / self.quantization_step))
        return quantized_coeffs
    def _dequantize(self, quantized_coeffs):
        coeffs = []
        for coeff in quantized_coeffs:
            if isinstance(coeff, tuple):
                # For detail coefficients
                dequantized_tuple = tuple(c * self.quantization_step for c in coeff)
                coeffs.append(dequantized_tuple)
            else:
                # For approximation coefficients
                coeffs.append(coeff * self.quantization_step)
        return coeffs
    def _entropy_encode(self, quantized_coeffs):
        # Convert coefficients to a flat array and store shapes
        flat_coeffs = []
        coeff_shapes = []
        for coeff in quantized_coeffs:
            if isinstance(coeff, tuple):
                shapes = []
                for c in coeff:
                    shapes.append(c.shape)
                    flat_coeffs.append(c.flatten())
                coeff_shapes.append(('tuple', shapes))
            else:
                coeff_shapes.append(('array', coeff.shape))
                flat_coeffs.append(coeff.flatten())
        flat_coeffs = np.concatenate(flat_coeffs)
        flat_coeffs = np.array(flat_coeffs, dtype=np.int32)
        # Use delta encoding
        deltas = np.diff(flat_coeffs)
        deltas = np.insert(deltas, 0, flat_coeffs[0])  # Store first value
        # Compress both shapes and coefficients
        compressed_shapes = zlib.compress(pickle.dumps(coeff_shapes))
        compressed_coeffs = zlib.compress(deltas.tobytes())
        # Combine with length prefixes
        combined = (struct.pack('I', len(compressed_shapes)) +
                    compressed_shapes +
                    struct.pack('I', len(compressed_coeffs)) +
                    compressed_coeffs)
        return combined
    def _entropy_decode(self, compressed_data):
        # Extract shapes length
        shapes_len = struct.unpack('I', compressed_data[:4])[0]
        offset = 4
        # Extract shapes data
        compressed_shapes = compressed_data[offset:offset + shapes_len]
        offset += shapes_len
        # Extract coeffs length
        coeffs_len = struct.unpack('I', compressed_data[offset:offset + 4])[0]
        offset += 4
        # Extract coeffs data
        compressed_coeffs = compressed_data[offset:offset + coeffs_len]
        # Decompress shapes
        coeff_shapes = pickle.loads(zlib.decompress(compressed_shapes))
        # Decompress coefficients
        deltas = np.frombuffer(zlib.decompress(compressed_coeffs), dtype=np.int32)
        # Reconstruct coefficients
        flat_coeffs = np.cumsum(deltas)
        # Rebuild coefficient structure
        quantized_coeffs = []
        current_pos = 0
        for shape_info in coeff_shapes:
            if shape_info[0] == 'tuple':
                shapes = shape_info[1]
                coeff_tuple = []
                for shape in shapes:
                    size = shape[0] * shape[1]
                    coeff = flat_coeffs[current_pos:current_pos + size].reshape(shape)
                    coeff_tuple.append(coeff)
                    current_pos += size
                quantized_coeffs.append(tuple(coeff_tuple))
            else:
                shape = shape_info[1]
                size = shape[0] * shape[1]
                coeff = flat_coeffs[current_pos:current_pos + size].reshape(shape)
                quantized_coeffs.append(coeff)
                current_pos += size
        return quantized_coeffs
    def _write_to_file(self, compressed_data, file_path, original_shape):
        with open(file_path, 'wb') as f:
            # Write header with original shape and parameters
            # Now includes channels: height, width, channels, levels, quantization_step
            f.write(struct.pack('IIIII',
                                original_shape[0],  # height
                                original_shape[1],  # width
                                original_shape[2],  # channels
                                self.levels,
                                self.quantization_step))
            # Write compressed data
            f.write(compressed_data)
    def _read_from_file(self, file_path):
        with open(file_path, 'rb') as f:
            # Read header (now 20 bytes: 5 unsigned integers)
            header = f.read(20)
            h, w, c, levels, q_step = struct.unpack('IIIII', header)
            self.levels = levels
            self.quantization_step = q_step
            # Read remaining data
            compressed_data = f.read()
            return compressed_data, (h, w, c)
    def calculate_psnr(self, original_path, reconstructed_path):
        """Tính PSNR giữa ảnh gốc và ảnh giải nén"""
        original = np.array(Image.open(original_path).convert('RGB'), dtype=np.float32)
        reconstructed = np.array(Image.open(reconstructed_path).convert('RGB'), dtype=np.float32)
        # Tính MSE (Mean Squared Error)
        mse = np.mean((original - reconstructed) ** 2)
        # Tránh chia cho 0
        if mse == 0:
            return float('inf')
        max_pixel = 255.0
        psnr = 20 * log10(max_pixel / sqrt(mse))
        return psnr
    def calculate_ssim(self, original_path, reconstructed_path):
        """Tính SSIM giữa ảnh gốc và ảnh giải nén"""
        original = np.array(Image.open(original_path).convert('L'), dtype=np.float32)  # Chuyển sang grayscale
        reconstructed = np.array(Image.open(reconstructed_path).convert('L'), dtype=np.float32)
        return ssim(original, reconstructed, data_range=255)
# Example usage
if __name__ == "__main__":
    # Create compressor
    jpeg2000 = JPEG2000(wavelet='db4', levels=4, quantization_step=30)
    # Compress
    input_image = "D:/Lenna_(test_image).png"
    compressed_file = "D:/compressed.j2k"
    jpeg2000.compress(input_image, compressed_file)
    print(f"Compressed {input_image} to {compressed_file}")
    # Get file sizes
    original_size = os.path.getsize(input_image)
    compressed_size = os.path.getsize(compressed_file)
    print(f"Original size: {original_size} bytes")
    print(f"Compressed size: {compressed_size} bytes")
    print(f"Compression ratio: {original_size / compressed_size:.2f}:1")
    # Decompress
    output_image = "D:/reconstructed.png"
    jpeg2000.decompress(compressed_file, output_image)
    print(f"Decompressed to {output_image}")
    # Calculate PSNR
    psnr_value = jpeg2000.calculate_psnr(input_image, output_image)
    print(f"PSNR: {psnr_value:.2f} dB")
    print(f"- SSIM: {jpeg2000.calculate_ssim(input_image, output_image):.4f}")
