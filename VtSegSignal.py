import os
import time
import cv2
import numpy as np
import base64
import shutil
from ultralytics import YOLO
from signalrcore.hub_connection_builder import HubConnectionBuilder

# Configuración de carpetas
camName = "IngOficina"
input_folder = "captured_images"
preprocess_folder = "preprocessFrames"
processing_folder = "processing_images"
detect_person_folder = "DetectPerson"
model = YOLO("yolov8x-seg.pt")
max_files = 25

# Configuración de SignalR
hub_url = "https://www.url del hub .com/HubAleko/testhub"
hub_connection = HubConnectionBuilder().with_url(hub_url).build()
hub_connection.start()

# Crear carpetas si no existen
for folder in [preprocess_folder, processing_folder, detect_person_folder]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Función para obtener la antepenúltima imagen
def get_antepenultimate_file(folder):
    files = [f for f in os.listdir(folder) if f.endswith(".jpg")]
    if len(files) > 2:
        files.sort(key=lambda x: os.path.getmtime(os.path.join(folder, x)))
        return os.path.join(folder, files[-3])
    return None

# Obtener las últimas imágenes de personas detectadas
def get_latest_images(folder, num_images):
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".jpg")]
    files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return files[:num_images]

# Procesar imagen con YOLOv8 (solo clase persona)
def process_image(image_path):
    image = cv2.imread(image_path)
    results = model.predict(image, classes=[0])
    output_image = image.copy()
    green_color = (0, 255, 0)
    person_count = 0

    for result in results:
        if result.masks is None:
            print("0 persons detected.")
            send_image_to_signalr(output_image, description="0 persons detected")
            send_images_to_signalr([output_image], camName, 0)
            return output_image

        masks = result.masks.data
        boxes = result.boxes.xyxy
        classes = result.boxes.cls

        for i, cls in enumerate(classes):
            person_count += 1
            mask = masks[i].cpu().numpy()
            mask = (mask > 0.5).astype(np.uint8)
            mask_resized = cv2.resize(mask, (output_image.shape[1], output_image.shape[0]))
            colored_mask = np.zeros_like(output_image)

            for c in range(3):
                colored_mask[:, :, c] = mask_resized * green_color[c]

            output_image = cv2.addWeighted(output_image, 1, colored_mask, 0.6, 0)
            x1, y1, x2, y2 = map(int, boxes[i].tolist())
            cv2.rectangle(output_image, (x1, y1), (x2, y2), green_color, 2)

            person_crop = image[y1:y2, x1:x2]
            person_path = os.path.join(detect_person_folder, f"person_{time.time()}.jpg")
            cv2.imwrite(person_path, person_crop)

    print(f"{person_count} persons detected.")
    latest_images = get_latest_images(detect_person_folder, person_count)
    send_images_to_signalr(latest_images, camName, person_count)
    send_image_to_signalr(output_image, description=f"{person_count} persons detected")
    return output_image

# Enviar una sola imagen a SignalR (mantener funcionalidad anterior)
def send_image_to_signalr(frame, description="Imagen procesada"):
    success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    hub_connection.send("SendImage", [image_base64, description])

# Enviar múltiples imágenes a SignalR (nueva funcionalidad)
def send_images_to_signalr(image_inputs, name_room, num_persons):
    # Si no se detecta ninguna persona, solo enviar el número 0
    if num_persons == 0:
        hub_connection.send("ListRoomPersons", [[], name_room, num_persons])
        return

    base64_images = []
    for image_input in image_inputs:
        # Verificar si es una ruta o una imagen numpy
        if isinstance(image_input, str):
            with open(image_input, "rb") as img_file:
                base64_images.append(base64.b64encode(img_file.read()).decode('utf-8'))
        elif isinstance(image_input, np.ndarray):
            _, buffer = cv2.imencode('.jpg', image_input, [cv2.IMWRITE_JPEG_QUALITY, 90])
            base64_images.append(base64.b64encode(buffer).decode('utf-8'))

    # Enviar las imágenes codificadas a SignalR
    hub_connection.send("ListRoomPersons", [base64_images, name_room, num_persons])

# Ciclo principal
while True:
    image_path = get_antepenultimate_file(input_folder)
    if image_path:
        print(f"Procesando: {image_path}")
        processed_image = process_image(image_path)
        preprocess_output_path = os.path.join(preprocess_folder, os.path.basename(image_path))
        cv2.imwrite(preprocess_output_path, processed_image)
        print(f"Imagen guardada en {preprocess_output_path}")

        # Mover la imagen a la carpeta de procesamiento
        processing_image_path = os.path.join(processing_folder, os.path.basename(image_path))
        shutil.move(preprocess_output_path, processing_image_path)
        print(f"Imagen movida a {processing_image_path}")

        # Limitar archivos en la carpeta
        if len(os.listdir(preprocess_folder)) > max_files:
            oldest_file = min(
                [os.path.join(preprocess_folder, f) for f in os.listdir(preprocess_folder)],
                key=os.path.getmtime
            )
            os.remove(oldest_file)
            print(f"Archivo eliminado: {oldest_file}")
    else:
        print("Esperando nuevas imágenes...")

    time.sleep(2)
