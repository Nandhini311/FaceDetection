import cv2
import matplotlib.pyplot as plt
import time
%matplotlib inline


def convertToRGB(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


#load test image
test1 = cv2.imread('/home/admins/Desktop/facedetect/data/test1.jpg')

gray_img = cv2.cvtColor(test1, cv2.COLOR_BGR2GRAY)

plt.imshow(gray_img, cmap='gray')


haar_face_cascade = cv2.CascadeClassifier('/home/admins/Desktop/facedetect/haarcascade_frontalface_alt.xml') 
faces = haar_face_cascade.detectMultiScale(gray_img)  
print('Faces found: ', len(faces))

def detect_faces(f_cascade, colored_img, scaleFactor = 1.1):
    img_copy = colored_img.copy()
    
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    faces = f_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=5);
    
    for(x,y,w,h) in faces:
        cv2.rectangle(img_copy, (x,y), (x+w, y+h), (0,255,0),2)
    return img_copy

test2 = cv2.imread('/home/admins/Desktop/facedetect/data/test2.jpg')


face_detected_img = detect_faces(haar_face_cascade, test2)
plt.imshow(convertToRGB(face_detected_img))

test3 = cv2.imread('/home/admins/Desktop/facedetect/data/test3.jpg')
face_detected_img1 = detect_faces(haar_face_cascade, test3, scaleFactor=1.1)
plt.imshow(convertToRGB(face_detected_img1))

#using lbpcascade training file
lbp_face_cascade = cv2.CascadeClassifier('/home/admins/Desktop/facedetect/lbpcascade_frontalface.xml')

#load test image
test4 = cv2.imread('/home/admins/Desktop/facedetect/data/test2.jpg')
faces_detected_img = detect_faces(lbp_face_cascade, test4)

plt.imshow(convertToRGB(faces_detected_img))

test5 = cv2.imread('/home/admins/Desktop/facedetect/data/test4.jpg')
faces_detected_img1 = detect_faces(lbp_face_cascade, test5)

plt.imshow(convertToRGB(faces_detected_img1))

#HAAR vs LBP results analysis

test1 = cv2.imread('/home/admins/Desktop/facedetect/data/test5.jpg')
test2 = cv2.imread('/home/admins/Desktop/facedetect/data/test6.jpg')

#test1
t1 = time.time()
haar_detected_img = detect_faces(haar_face_cascade, test1)

t2 = time.time()
dt1 = t2-t1
print(dt1)

#time calculaytion for lbp classifier
t3 = time.time()
lbp_detected_img = detect_faces(lbp_face_cascade, test1)
t4 = time.time()

dt2 = t4-t3
print(dt2)

#show HAAR image
f, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5))
ax1.set_title('Haar Detection time:' + str(round(dt1,3)) + 'secs')
ax1.imshow(convertToRGB(haar_detected_img))

#show LBP image
ax2.set_title('LBP Detection time:' + str(round(dt2,3))+'secs')
ax2.imshow(convertToRGB(lbp_detected_img))

#test2
t5 = time.time()
haar_detected_img = detect_faces(haar_face_cascade, test2)

t6 = time.time()
dt3 = t6-t5
print(dt3)

#time calculaytion for lbp classifier
t7 = time.time()
lbp_detected_img = detect_faces(lbp_face_cascade, test2)
t8 = time.time()

dt4 = t8-t7
print(dt4)

#show HAAR image
f, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5))
ax1.set_title('Haar Detection time:' + str(round(dt3,3)) + 'secs')
ax1.imshow(convertToRGB(haar_detected_img))

#show LBP image
ax2.set_title('LBP Detection time:' + str(round(dt4,3))+'secs')
ax2.imshow(convertToRGB(lbp_detected_img))
