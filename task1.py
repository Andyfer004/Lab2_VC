import cv2
import numpy as np
import os

img_path = "images/periodic_noise.jpg"
out_dir = "outputs/task1"
os.makedirs(out_dir, exist_ok=True)

img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise Exception(f"No se pudo leer la imagen: {img_path}")

h, w = img.shape
cy, cx = h // 2, w // 2

img_f = img.astype(np.float32)
dft = cv2.dft(img_f, flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft, axes=(0, 1))

mag = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
mag_log = np.log(mag + 1)
mag_vis = cv2.normalize(mag_log, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

mask2d = np.ones((h, w), dtype=np.uint8)

peaks = [
    (272, 269),
    (269, 272),
    (280, 277),
]

r = 10

for x, y in peaks:
    cv2.circle(mask2d, (x, y), r, 0, -1)
    cv2.circle(mask2d, (2*cx - x, 2*cy - y), r, 0, -1)

mask = np.repeat(mask2d[:, :, None], 2, axis=2)

filtered = dft_shift * mask

back_shift = np.fft.ifftshift(filtered, axes=(0, 1))
img_back = cv2.idft(back_shift)
img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

restored = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

cv2.imwrite(os.path.join(out_dir, "01_original.png"), img)
cv2.imwrite(os.path.join(out_dir, "02_spectrum_log.png"), mag_vis)
cv2.imwrite(os.path.join(out_dir, "03_mask.png"), (mask2d * 255).astype(np.uint8))
cv2.imwrite(os.path.join(out_dir, "04_restored.png"), restored)

print("Listo.")