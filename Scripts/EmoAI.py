# Importar las bibliotecas necesarias
import tkinter as tk  # Biblioteca para la interfaz gráfica
# Clases para manejar imágenes y diálogos de archivo
from tkinter import PhotoImage, filedialog
import cv2  # OpenCV para el procesamiento de imágenes
from PIL import Image, ImageTk  # PIL para manipular imágenes
from keras.models import model_from_json  # Cargar modelos de Keras
import numpy as np  # Numpy para operaciones numéricas

# Cargar la arquitectura del modelo pre-entrenado desde un archivo JSON
json_file = open("../Model/emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)

# Cargar los pesos del modelo pre-entrenado
model.load_weights("../Model/emotiondetector.h5")

# Definir etiquetas para las clases de emociones
labels = {0: 'Enojado', 1: 'Disgustado', 2: 'Miedo',
          3: 'Feliz', 4: 'Neutral', 5: 'Triste', 6: 'Sorprendido'}

# Cargar el clasificador de cascada Haar para la detección de rostros
face_cascade = cv2.CascadeClassifier(
    '../Model/haarcascade_frontalface_default.xml')

# Cargar imágenes de emociones
emotion_images = {
    'Enojado': cv2.imread('../Pictures/angry.png'),
    'Disgustado': cv2.imread('../Pictures/disgust.png'),
    'Miedo': cv2.imread('../Pictures/fear.png'),
    'Feliz': cv2.imread('../Pictures/happy.png'),
    'Neutral': cv2.imread('../Pictures/neutral.png'),
    'Triste': cv2.imread('../Pictures/sad.png'),
    'Sorprendido': cv2.imread('../Pictures/surprised.png')
}

# Funcion para destruir la ventana de notificacion


def ok_button_click():
    notification.destroy()

# Función para extraer características de la imagen


def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Función para detectar rostros y emociones


def detect_faces(frame):
    # Convertir el fotograma a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar rostros en el fotograma
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Verificar si se detectaron rostros
    if len(faces) > 0:
        # Para cada rostro detectado, realizar la detección de emociones
        for (x, y, w, h) in faces:
            # Extraer la región de interés (ROI) que contiene el rostro
            face_roi = gray[y:y+h, x:x+w]

            # Redimensionar la imagen del rostro a la forma requerida (48x48)
            face_roi = cv2.resize(face_roi, (48, 48))

            # Extraer características de la imagen del rostro
            img = extract_features(face_roi)

            # Realizar una predicción usando el modelo entrenado
            pred = model.predict(img)

            # Obtener la etiqueta predicha para la emoción
            emotion_label = labels[pred.argmax()]

            # Dibujar un rectángulo alrededor del rostro detectado
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Mostrar la emoción predicha cerca del rostro detectado
            # Obtener el tamaño del texto
            text_size = cv2.getTextSize(
                f'Emoción: {emotion_label}', cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 2)

            # Calcular las coordenadas de posición del texto
            # Ajustar la posición horizontal del texto para evitar que se salga del borde izquierdo
            text_x = max(x - 10, 0)
            # Ajustar la posición vertical del texto para evitar que se salga del borde superior
            text_y = max(y - 10, text_size[0][1] + 10)

            # Si el texto se dibuja muy cerca del borde derecho o inferior de la imagen, ajustar su posición
            if text_x + text_size[0][0] > frame.shape[1]:
                text_x = max(frame.shape[1] - text_size[0][0], 0)
            if text_y + text_size[0][1] > frame.shape[0]:
                text_y = max(frame.shape[0] - text_size[0][1], 0)

            # Dibujar el texto de la emoción predicha cerca del rostro detectado
            cv2.putText(frame, f'Emocion: {emotion_label}', (text_x, text_y),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

            # Mostrar una imagen en la esquina inferior izquierda de la pantalla
            if emotion_label in emotion_images:
                emotion_img = emotion_images[emotion_label]
                # Redimensionar la imagen de la emoción para que coincida con el área de superposición
                emotion_img_resized = cv2.resize(emotion_img, (100, 100))
                # Ajustar las coordenadas del área de superposición para que coincidan con las dimensiones de la imagen redimensionada
                roi_y1, roi_y2 = 250, 250 + emotion_img_resized.shape[0]
                roi_x1, roi_x2 = 10, 10 + emotion_img_resized.shape[1]

               # Asegurarse de que la región de superposición tenga las mismas dimensiones que la imagen redimensionada
                if roi_y2 - roi_y1 > 0 and roi_x2 - roi_x1 > 0:
                    # Asegurar que las coordenadas de la región de superposición estén dentro de los límites de la imagen
                    if roi_y2 <= frame.shape[0] and roi_x2 <= frame.shape[1]:
                        frame[roi_y1:roi_y2, roi_x1:roi_x2] = emotion_img_resized[:roi_y2 -
                                                                                  roi_y1, :roi_x2 - roi_x1]
                    else:
                        print(
                            "Error: Las coordenadas de la región de superposición exceden los límites de la imagen")
                else:
                    print(
                        "Error: Las dimensiones de la región de superposición no son válidas")
    return frame
# Función para cargar una imagen desde el explorador de archivos


def load_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_with_faces = detect_faces(img)

        # Mostrar la imagen en una nueva ventana junto con la emoción detectada
        show_image_with_emotion(img_with_faces)


# Función para mostrar la imagen junto con la emoción detectada en una nueva ventana
def show_image_with_emotion(img):
    # Crear una nueva ventana
    image_window = tk.Toplevel(window)
    image_window.title("Detección de Emociones")

    # Redimensionar la ventana de acuerdo al tamaño de la imagen
    image_window.geometry(f"{img.shape[1]}x{img.shape[0]}")

    # Mostrar la imagen
    img = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=img)
    label_image = tk.Label(image_window, image=imgtk)
    label_image.image = imgtk
    label_image.pack()


# Crear la ventana principal
window = tk.Tk()
window.title("EmoAI")
window.geometry("500x425")
window.overrideredirect(True)
window.configure(bg="#5a32e2")

# Marco para la cámara
border_frame = tk.Frame(window, bg="white")
border_frame.place(relx=0.01, rely=0.01, relwidth=0.981, relheight=0.982)

# Barra de título
banner = tk.Frame(window, bg="#5a32e2")
banner.pack(side="top", fill="x")

# Logotipo
logo_img = PhotoImage(file="../Pictures/logo.png").subsample(8, 8)
label_logo = tk.Label(banner, image=logo_img, bg="#5C32E2")
label_logo.image = logo_img
label_logo.pack(side="left", padx=15, pady=10)

# Texto del logotipo
text_Logo = tk.Label(banner, text="EmoAI", bg="#5a32e2",
                     font=('Arial', 14, 'bold'), fg="white", height=2)
text_Logo.pack(side="left", fill="x")

# Botón de cierre
close_img = PhotoImage(file="../Pictures/close.png").subsample(19, 19)
close_button = tk.Button(banner, image=close_img, bg="#5a32e2",
                         activebackground="#5a32e2", bd=0, command=window.destroy)
close_button.image = close_img
close_button.pack(side="right", padx=15, pady=10)

# Botón para seleccionar imagen
select_img = PhotoImage(file="../Pictures/upload.png").subsample(3, 3)
select_button = tk.Button(banner, image=select_img, text="Seleccionar imagen", bg="#5a32e2",
                          activebackground="#5a32e2", bd=0, command=load_image)
select_button.image = select_img
select_button.pack(side="right", padx=15, pady=10)

# Etiqueta para mostrar la cámara
label_camara = tk.Label(border_frame)
label_camara.place(relx=0.5, rely=0.05, anchor=tk.N)

# Función para mover la ventana
xwin = 0
ywin = 0


def iniciar_mover_ventana(event):
    global xwin, ywin
    xwin, ywin = event.x, event.y


def mover_ventana(event):
    x, y = event.x_root - xwin, event.y_root - ywin
    window.geometry(f"+{x}+{y}")


banner.bind("<Button-1>", iniciar_mover_ventana)
banner.bind("<B1-Motion>", mover_ventana)

# Función para capturar fotogramas de la cámara y actualizar la etiqueta


def update_frame():
    ret, frame = cap.read()
    if ret:
        # Redimensionar el fotograma para que se ajuste a la etiqueta
        frame = cv2.resize(frame, (480, 360))
        frame_with_faces = detect_faces(frame)
        frame = cv2.cvtColor(frame_with_faces, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        label_camara.imgtk = imgtk
        label_camara.configure(image=imgtk)
    label_camara.after(10, update_frame)


# Inicializar la cámara
cap = cv2.VideoCapture(0)
label_camara = tk.Label(border_frame)
label_camara.pack(pady=(50, 1))
notification = tk.Frame(border_frame, bg="#5a32e2")
notification.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
notification.config(highlightbackground="white", highlightthickness=2)
image_path = "../Pictures/logo.png"
image = PhotoImage(file=image_path).subsample(3, 3)
image_label = tk.Label(notification, image=image, bg="#5a32e2")
image_label.image = image
image_label.pack(side="top", padx=15, pady=10)
text_notification = tk.Label(notification, text="¡Hola! Bienvenido a EmoAI, por favor evita moverte mucho.",
                             bg="#5a32e2", font=('Arial', 20, 'bold'), fg="white", height=3, wraplength=350)
text_notification.pack(side="top", fill="x", padx=15)
ok_button = tk.Button(notification, text="Ok", bg="white", font=('Arial', 13, 'bold'), fg="#5a32e2",
                      height=1, width=5, bd=0, relief=tk.FLAT, cursor="hand2", pady=5, borderwidth=2, command=ok_button_click)
ok_button.pack(side="top", pady=(0, 10))


# Actualizar fotogramas de la cámara
update_frame()

# Ejecutar la ventana
window.mainloop()
