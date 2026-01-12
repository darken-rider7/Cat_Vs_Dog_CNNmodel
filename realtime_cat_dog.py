import cv2
import numpy as np
from keras.models import load_model

model = load_model(
    r"D:\python\Aiml\projects\catanddog_CNN\cat_dog_cnn.h5"
)

img_size = (120, 120) 

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
   
    img = cv2.resize(frame, img_size)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img, verbose=0)
    cv2.flip(frame, 1, frame)
    if pred[0][0] > 0.5:
        label = "Dog"
        color = (0, 255, 0)
    else:
        label = "Cat"
        color = (255, 0, 0)

    cv2.putText(frame, label, (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Cat vs Dog Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
