import cv2
import numpy as np
import os

# -----------------------------
# Configuración
# -----------------------------
img_path = "images/textile_defect.jpg"
out_dir = "outputs/task3"
os.makedirs(out_dir, exist_ok=True)

img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise Exception(f"No se pudo leer la imagen: {img_path}")

h, w = img.shape
cy, cx = h // 2, w // 2

print("TASK 3: Detección de Rasgaduras en Tela (FFT + Morfología)")
print(f"Imagen: {img_path}")
print(f"Dimensiones: {w}x{h}")

# ============================================================
# PASO 1) Fourier: FFT -> Filtro -> IFFT (Supresión de textura)
# ============================================================
# Convertir a float32 para DFT
img_f = img.astype(np.float32)

# DFT y shift al centro
dft = cv2.dft(img_f, flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft, axes=(0, 1))

# Magnitud para visualizar espectro
mag = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
mag_log = np.log(mag + 1)
mag_vis = cv2.normalize(mag_log, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# -----------------------------
# Filtro PASA-BAJAS gaussiano
# -----------------------------
D0 = 30  # puedes ajustar 20-50

u = np.arange(w)
v = np.arange(h)
u, v = np.meshgrid(u - cx, v - cy)
D = np.sqrt(u**2 + v**2)

H = np.exp(-(D**2) / (2 * (D0**2)))  # PB gaussiano
H_vis = cv2.normalize(H, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

H_complex = np.repeat(H[:, :, None], 2, axis=2)
filtered_dft = dft_shift * H_complex

# IFFT
back_shift = np.fft.ifftshift(filtered_dft, axes=(0, 1))
img_back = cv2.idft(back_shift)
img_filtered = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

img_suavizada = cv2.normalize(img_filtered, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# ============================================================
# PASO 2) Thresholding (Segmentación)
# Queremos SOLO la rasgadura como blanco
# ============================================================
mu = img_suavizada.mean()
sigma = img_suavizada.std()

# k controla cuán "estricto" es el umbral
# Si te queda muy poca detección, baja k (ej 1.5)
# Si te salen falsos positivos, sube k (ej 2.2)
k = 2.0
T = int(np.clip(mu + k * sigma, 0, 255))

_, mask_preliminar = cv2.threshold(img_suavizada, T, 255, cv2.THRESH_BINARY)

# ============================================================
# PASO 3) Morfología (Refinamiento)
# ============================================================

# 3.1 Apertura: elimina puntitos aislados (ruido residual)
kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
mask_limpia = cv2.morphologyEx(mask_preliminar, cv2.MORPH_OPEN, kernel_open, iterations=1)

# 3.2 Cierre: conecta pedazos y rellena huecos internos
kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
mask_conectada = cv2.morphologyEx(mask_limpia, cv2.MORPH_CLOSE, kernel_close, iterations=1)

# 3.3 Mantener SOLO el componente conectado más grande (rasgadura)
contours, _ = cv2.findContours(mask_conectada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

mask_final = np.zeros_like(mask_conectada)
if len(contours) > 0:
    biggest = max(contours, key=cv2.contourArea)
    cv2.drawContours(mask_final, [biggest], -1, 255, -1)

# 3.4 Dilatación leve final para cubrir bien la rasgadura
kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
mask_final = cv2.dilate(mask_final, kernel_dilate, iterations=1)

# ============================================================
# Guardar resultados
# ============================================================
print("\nGuardando resultados...")

cv2.imwrite(os.path.join(out_dir, "01_original.png"), img)
cv2.imwrite(os.path.join(out_dir, "02_espectro_original.png"), mag_vis)
cv2.imwrite(os.path.join(out_dir, "03_filtro_gaussiano.png"), H_vis)
cv2.imwrite(os.path.join(out_dir, "04_imagen_suavizada.png"), img_suavizada)
cv2.imwrite(os.path.join(out_dir, "05_mask_preliminar.png"), mask_preliminar)
cv2.imwrite(os.path.join(out_dir, "06_mask_final.png"), mask_final)

# Overlay (rojo sobre la imagen original)
img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
overlay = img_color.copy()
overlay[mask_final > 0] = [0, 0, 255]
resultado_overlay = cv2.addWeighted(img_color, 0.7, overlay, 0.3, 0)
cv2.imwrite(os.path.join(out_dir, "07_resultado_overlay.png"), resultado_overlay)

# Pipeline completo (8-panel)
fig_h, fig_w = 300, 300

def resize_for_display(image, size=(fig_w, fig_h)):
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

row1 = np.hstack([
    resize_for_display(img),
    resize_for_display(mag_vis),
    resize_for_display(H_vis),
    resize_for_display(img_suavizada)
])

row2 = np.hstack([
    resize_for_display(mask_preliminar),
    resize_for_display(mask_final),
    resize_for_display(cv2.cvtColor(resultado_overlay, cv2.COLOR_BGR2GRAY)),
    resize_for_display(np.zeros((h, w), dtype=np.uint8))
])

comparacion = np.vstack([row1, row2])

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
thickness = 1
color = 200

labels_row1 = ["Original", "Espectro", "Filtro PB", "Suavizada"]
labels_row2 = ["Mask Prelim", "Mask Final", "Overlay", ""]

for i, label in enumerate(labels_row1):
    cv2.putText(comparacion, label, (i * fig_w + 10, 25), font, font_scale, color, thickness)

for i, label in enumerate(labels_row2):
    cv2.putText(comparacion, label, (i * fig_w + 10, fig_h + 25), font, font_scale, color, thickness)

cv2.imwrite(os.path.join(out_dir, "08_pipeline_completo.png"), comparacion)

print(f"\nImágenes guardadas en: {out_dir}/")
print("  - 01_original.png")
print("  - 02_espectro_original.png")
print("  - 03_filtro_gaussiano.png")
print("  - 04_imagen_suavizada.png")
print("  - 05_mask_preliminar.png")
print("  - 06_mask_final.png  (RESULTADO FINAL ✅)")
print("  - 07_resultado_overlay.png")
print("  - 08_pipeline_completo.png")
print("\nListo ✅")