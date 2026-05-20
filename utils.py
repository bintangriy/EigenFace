import os
import cv2
import numpy as np

IMG_SIZE = 128

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def preprocess_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5
    )

    if len(faces) == 0:
        return None

    x, y, w, h = faces[0]

    face = gray[y:y+h, x:x+w]
    face = cv2.equalizeHist(face)
    face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))

    return face.flatten()

def load_images_from_folder(folder):
    images = []
    labels = []
    label_names = []
    color_images = []
    label_map = {}

    label_index = 0

    for person_name in os.listdir(folder):
        person_path = os.path.join(folder, person_name)

        if not os.path.isdir(person_path):
            continue

        if person_name not in label_map:
            label_map[person_name] = label_index
            label_names.append(person_name)
            label_index += 1

        for filename in os.listdir(person_path):
            file_path = os.path.join(person_path, filename)

            img_color = cv2.imread(file_path)

            if img_color is None:
                continue

            processed = preprocess_face(img_color)

            if processed is None:
                continue

            img_display = cv2.resize(img_color, (IMG_SIZE, IMG_SIZE))

            images.append(processed)
            color_images.append(
                cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)
            )
            labels.append(label_map[person_name])

    return np.array(images).T, labels, label_names, color_images