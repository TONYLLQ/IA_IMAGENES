import numpy as np
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import cv2

longitud, altura = 150, 150
modelo = 'CNN/modelo/modelo.keras'
pesos_modelo = 'CNN/modelo/pesos.weights.h5'
cnn = load_model(modelo)
cnn.load_weights(pesos_modelo)

def predict(array):
    x = img_to_array(array)
    x = np.expand_dims(x, axis=0)
    array = cnn.predict(x)
    result = array[0]
    answer = np.argmax(result)
    return answer

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error al abrir la cámara")
    exit()

threshold = 0.5

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al capturar el fotograma")
        break

    frame = cv2.resize(frame, (longitud, altura)) # Asegúrate de que la imagen tenga la forma correcta
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
    
    answer=predict(frame)

   

    frame = cv2.resize(frame, (500, 500))
    if answer == 0:
        cv2.putText(frame, "OSO", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    elif answer==1:
        cv2.putText(frame, "GATO", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    elif answer==2:
        cv2.putText(frame, "PERRO", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    elif answer==3:
        cv2.putText(frame, "ELEFANTE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    elif answer==4:
        cv2.putText(frame, "CABALLO", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    elif answer==5:
        cv2.putText(frame, "LEON", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    elif answer==6:
        cv2.putText(frame, "LEON", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    elif answer==7:
        cv2.putText(frame, "TIGRE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    elif answer==8:
        cv2.putText(frame, "ZEBRA", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    
    cv2.imshow('Detección de Objetos', frame)

    # Salir del bucle si se presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la captura de video y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()
