import cv2
import os
import time

print("start GetCam VT...")

def limit_images(folder, max_files=50):
    """Mantiene un número máximo de imágenes en la carpeta eliminando las más antiguas."""
    try:
        files = sorted(
            (os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))),
            key=os.path.getmtime
        )
        if len(files) > max_files:
            for file_to_delete in files[:-max_files]:
                os.remove(file_to_delete)
    except Exception as e:
        print(f"Error al limitar archivos en la carpeta '{folder}': {e}")

def connect_camera(camera_url):
    """Intenta conectar con la cámara y retorna el objeto VideoCapture."""
    cap = cv2.VideoCapture(camera_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 200)
    if not cap.isOpened():
        print("Error: No se pudo conectar a la cámara.")
    return cap

# Configuración de la cámara
camera_url = "rtsp://admin:usuario@ip:puerto/stream"
output_folder = "captured_images"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

cap = connect_camera(camera_url)

while True:
    if not cap.isOpened():
        print("Intentando reconectar con la cámara...")
        cap = connect_camera(camera_url)
        time.sleep(2)
        continue  # Reintentar sin procesar la imagen

    # Descartar frames obsoletos
    for _ in range(5):  
        cap.grab()

    ret, frame = cap.read()
    if ret:
        resized_frame = cv2.resize(frame, (2688, 1520))
        timestamp = int(time.time())
        filename = os.path.join(output_folder, f"frame_{timestamp}.jpg")
        cv2.imwrite(filename, resized_frame)
        limit_images(output_folder, max_files=50)
    else:
        print("Error: No se pudo leer el frame de la cámara. Intentando reconectar...")
        cap.release()  # Cerrar conexión anterior
        cap = connect_camera(camera_url)  # Intentar reconectar
        time.sleep(2)  # Esperar antes de reintentar para evitar bucles rápidos

cap.release()
