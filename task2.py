import cv2
import numpy as np
import os


img_path = "images/fingerprint_noisy.png"
out_dir = "outputs/task2"
os.makedirs(out_dir, exist_ok=True)

# 1. Cargar la imagen y asegurar que sea binaria
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise Exception(f"No se pudo leer la imagen: {img_path}")


_, img_binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# 2. Eliminar ruido sal (puntos blancos en valles negros) usando APERTURA
kernel_apertura = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
img_sin_ruido = cv2.morphologyEx(img_binary, cv2.MORPH_OPEN, kernel_apertura)

# 3. Conectar grietas en las crestas usando CIERRE
kernel_cierre = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
img_final = cv2.morphologyEx(img_sin_ruido, cv2.MORPH_CLOSE, kernel_cierre)

# 4. Guardar las imágenes resultantes
cv2.imwrite(os.path.join(out_dir, "01_original.png"), img_binary)
cv2.imwrite(os.path.join(out_dir, "02_sin_ruido_apertura.png"), img_sin_ruido)
cv2.imwrite(os.path.join(out_dir, "03_final_cierre.png"), img_final)

# Crear una imagen comparativa lado a lado
h, w = img_binary.shape
comparacion = np.zeros((h, w * 3), dtype=np.uint8)
comparacion[:, 0:w] = img_binary
comparacion[:, w:2*w] = img_sin_ruido
comparacion[:, 2*w:3*w] = img_final

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(comparacion, "Original", (10, 30), font, 0.7, 128, 2)
cv2.putText(comparacion, "Apertura (sin ruido)", (w + 10, 30), font, 0.7, 128, 2)
cv2.putText(comparacion, "Cierre (conectado)", (2*w + 10, 30), font, 0.7, 128, 2)

cv2.imwrite(os.path.join(out_dir, "04_comparacion.png"), comparacion)



print("\n" + "-" * 40)
print(f"Imágenes guardadas en: {out_dir}/")
print("  - 01_original.png")
print("  - 02_sin_ruido_apertura.png")
print("  - 03_final_cierre.png")
print("  - 04_comparacion.png")
print("\nListo.")
