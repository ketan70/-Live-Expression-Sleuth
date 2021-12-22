import cv2
import numpy as np 
cap = cv2.VideoCapture(0)
import tensorflow as tf
from tensorflow.keras.models import load_model

model = tf.keras.models.load_model('ketan72')

while(True):
    check, frame = cap.read()
    
    r = cv2.resize(frame,(140,140))
    x=[]
    x.append(r)

    g = np.array(x)
    y = model.predict(g)

    a =np.argmax(y)
    i_name = ["angry","fear","happy"]

    cv2.putText(frame,i_name[a],(75,75),cv2.FONT_HERSHEY_SIMPLEX,1,(34,45,246),2)

  




    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()