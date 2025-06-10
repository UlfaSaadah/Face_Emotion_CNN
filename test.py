# cv2: Mengolah gambar dan video (OpenCV).
# numpy: Mengelola data array.
# load_model: Memuat model CNN yang telah dilatih.
import cv2
import numpy as np
from keras.models import load_model

# Model CNN yang telah dilatih.
model=load_model('model_file_30epochs.h5')

# Mengaktifkan webcam (default kamera).
video=cv2.VideoCapture(0)

# Menggunakan Haar Cascade Classifier untuk mendeteksi wajah.
faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Memetakan model keluaran angka ke label ekspresi.
labels_dict={0:'Marah',1:'Jijik', 2:'Takut', 3:'Senang',4:'Netral',5:'Sedih',6:'Terkejut'}

# Membaca frame dari webcam
while True:
    ret,frame=video.read()
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces= faceDetect.detectMultiScale(gray, 1.3, 3)

    # Ekstrak wajah, normalisasi, prediksi ekspresi dengan model CNN.
    for x,y,w,h in faces:
        sub_face_img=gray[y:y+h, x:x+w]
        resized=cv2.resize(sub_face_img,(48,48))
        normalize=resized/255.0
        reshaped=np.reshape(normalize, (1, 48, 48, 1))
        result=model.predict(reshaped)
        label=np.argmax(result, axis=1)[0]
        print(label)
        # Menampilkan kotak dan label ekspresi di atas wajah yang terdeteksi.
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),2)
        cv2.rectangle(frame,(x,y-40),(x+w,y),(50,50,255),-1)
        cv2.putText(frame, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        
    cv2.imshow("Frame",frame)
    k=cv2.waitKey(1)
    if k==ord('q'):
        break

video.release()
cv2.destroyAllWindows()