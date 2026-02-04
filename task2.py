import cv2
import numpy as np
import os

img_path = "images/fingerprint_noisy.png"
out_dir = "outputs/task2"
os.makedirs(out_dir, exist_ok=True)

# 1) Cargar la imagen y asegurar binarización
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise Exception(f"No se pudo leer la imagen: {img_path}")

# Suavizado leve para estabilizar Otsu
img_blur = cv2.GaussianBlur(img, (3, 3), 0)

# Binarización (invertida) con Otsu:
# - objetivo: crestas blancas, fondo negro
_, bin_inv = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# 2) Eliminar ruido sal usando APERTURA (erosión -> dilatación)
kernel_apertura = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
sin_ruido = cv2.morphologyEx(bin_inv, cv2.MORPH_OPEN, kernel_apertura)

# 3) Conectar grietas usando CIERRE (dilatación -> erosión)
kernel_cierre = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
img_final = cv2.morphologyEx(sin_ruido, cv2.MORPH_CLOSE, kernel_cierre)

# -----------------------------
# DEMOSTRACIÓN: orden inverso
# (Cierre -> Apertura)
# -----------------------------
inv_cierre = cv2.morphologyEx(bin_inv, cv2.MORPH_CLOSE, kernel_cierre)
inv_final = cv2.morphologyEx(inv_cierre, cv2.MORPH_OPEN, kernel_apertura)

# 4) Guardar resultados del pipeline principal
cv2.imwrite(os.path.join(out_dir, "00_grayscale.png"), img)
cv2.imwrite(os.path.join(out_dir, "01_binaria.png"), bin_inv)
cv2.imwrite(os.path.join(out_dir, "02_sin_ruido_apertura.png"), sin_ruido)
cv2.imwrite(os.path.join(out_dir, "03_final_cierre.png"), img_final)

# Comparación principal: Binaria | Apertura | Cierre
h, w = bin_inv.shape
comparacion = np.zeros((h, w * 3), dtype=np.uint8)
comparacion[:, 0:w] = bin_inv
comparacion[:, w:2*w] = sin_ruido
comparacion[:, 2*w:3*w] = img_final

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(comparacion, "Binaria", (10, 30), font, 0.7, 255, 2)
cv2.putText(comparacion, "Apertura", (w + 10, 30), font, 0.7, 255, 2)
cv2.putText(comparacion, "Cierre", (2*w + 10, 30), font, 0.7, 255, 2)

cv2.imwrite(os.path.join(out_dir, "04_comparacion.png"), comparacion)

# 5) Guardar resultados del orden inverso
cv2.imwrite(os.path.join(out_dir, "05_inv_cierre.png"), inv_cierre)
cv2.imwrite(os.path.join(out_dir, "06_inv_final_apertura.png"), inv_final)

# Comparación inversa: Binaria | Cierre primero | Luego apertura
comparacion_inv = np.zeros((h, w * 3), dtype=np.uint8)
comparacion_inv[:, 0:w] = bin_inv
comparacion_inv[:, w:2*w] = inv_cierre
comparacion_inv[:, 2*w:3*w] = inv_final

cv2.putText(comparacion_inv, "Binaria", (10, 30), font, 0.7, 255, 2)
cv2.putText(comparacion_inv, "Cierre primero", (w + 10, 30), font, 0.7, 255, 2)
cv2.putText(comparacion_inv, "Luego apertura", (2*w + 10, 30), font, 0.7, 255, 2)

cv2.imwrite(os.path.join(out_dir, "07_comparacion_inversa.png"), comparacion_inv)

print("\n" + "-" * 50)
print(f"Imágenes guardadas en: {out_dir}/")
print("Pipeline principal:")
print("  - 00_grayscale.png")
print("  - 01_binaria.png")
print("  - 02_sin_ruido_apertura.png")
print("  - 03_final_cierre.png")
print("  - 04_comparacion.png")
print("\nOrden inverso (demostración):")
print("  - 05_inv_cierre.png")
print("  - 06_inv_final_apertura.png")
print("  - 07_comparacion_inversa.png")
print("-" * 50)
print("Listo ✅")