import os
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
#from retinaface import RetinaFace #crop faces
VIDEOS_FOLDER = ".\Data\Videos"
FACE_IMAGES_FOLDER = ".\Data\Face_images"
CROPPED_IMAGE_FOLDER = ".\Data\Cropped_images"
face_cascade = cv2.CascadeClassifier(".\haarcascade\haarcascade_frontalface_alt.xml")
side_face_cascade = cv2.CascadeClassifier(".\haarcascade\haarcascade_profileface.xml")###
#size=350 x 350 resize필요
#무조건 가로영상

class Make_datas:
    global cap
    def __init__(self):
        #create basic folders
        os.makedirs(VIDEOS_FOLDER, exist_ok=True)
        os.makedirs(FACE_IMAGES_FOLDER, exist_ok=True)
        os.makedirs(CROPPED_IMAGE_FOLDER, exist_ok=True)
        self.face_size = 224

    def detect_face(self,frame):
        faces = face_cascade.detectMultiScale(frame, scaleFactor=1.05, minNeighbors=5, minSize=(100, 100),
                                             flags=cv2.CASCADE_SCALE_IMAGE)
        print(faces)
        # 얼굴에 사각형 표시
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y-int(y*0.2)), (x + w, y + int(1.1*h)), (255, 0, 0), 2)
        return frame

    def film_video(self):
        global cap
        naming_folder = str(input("폴더 이름을 입력하시오: "))
        save_video_path = os.path.join(VIDEOS_FOLDER, naming_folder)
        print(save_video_path)
        os.makedirs(save_video_path, exist_ok=True)
        try:
            print("카메라 구동")
            cap = cv2.VideoCapture(0)
        except:
            print("카메라 구동실패")
            return
        width = int(cap.get(3))  # frame_width
        height = int(cap.get(4))  # frame_height

        fourcc = cv2.VideoWriter_fourcc(*'DIVX')                                          # fps     size
        out = cv2.VideoWriter(os.path.join(save_video_path, naming_folder + ".avi"), fourcc, 20.0, (width, height))
        #.\Data\Videos\yuna_hong\yuna_hong1.avi
        while (True):
            ret, frame = cap.read()
            if not ret:
                print("비디오 읽기 오류")
                break
            frame = self.detect_face(frame)
            cv2.imshow('video', frame)
            out.write(frame)
            if (cv2.waitKey(1) == 27):
                print('녹화 종료')
                break
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    def get_frames(self, video_file, save_folder):
        print("extract",video_file)
        vdo = cv2.VideoCapture(video_file)
        print("vdo",vdo)
        #frame_count, width, height, fps
        length, width, height, fps = int(vdo.get(7)), int(vdo.get(3)),int(vdo.get(4)),int(vdo.get(5))
        print(f"length: {length}, w x h: {width} x {height}, fps: {fps}\n")
        frame_count = 0
        while (vdo.isOpened()):
            ret, frame = vdo.read()
            if not ret:
                print("비디오 읽기 오류")
                vdo.release()
                cv2.destroyAllWindows()
                break

            if frame_count%15 == 0:
                cv2.imshow('video', frame)
                fullimg = os.path.basename(video_file).replace(".", "_") + "_" + str(frame_count) + ".png"
                #IU.avi IU_avi_11.png
                fullimg = os.path.join(save_folder, fullimg)
                #C:\Users\USER\Desktop\Facedlib\MakeData\Data\Face_images\IU \ IU_avi_11.png
                print("imgfile", fullimg)
                #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                cv2.imwrite(fullimg, frame)
                frame_count += 1
            else:
                frame_count += 1
                continue

            if cv2.waitKey(5) == 27:  # ESC key press
                break
            if vdo.get(cv2.CAP_PROP_POS_FRAMES) == vdo.get(cv2.CAP_PROP_FRAME_COUNT):
                break
'''
    def extract_faces(self, save_folder):
        pp=os.path.join(CROPPED_IMAGE_FOLDER,save_folder.split('\\')[-1])
        #C: \ Users \ USER \ Desktop\Facedlib\MakeData\Data\ Cropped_images \ IU
        os.makedirs(pp, exist_ok=True)
        print("extract_Faces",pp)
        folders = list(glob.iglob(os.path.join(save_folder, '*.*')))

        for img_path in folders:
            resp = RetinaFace.extract_faces(img_path=img_path, align=True)
            for img in resp:
                print(img.shape)
                img = cv2.resize(img, (300,420))
                cv2.imwrite(pp+"\\"+img_path.split("\\")[-1], img)
            print(save_folder.split('\\')[-1],"완료")
'''

def main(extractor):
    #extractor.film_video() #filming video

    folders = list(glob.iglob(os.path.join(VIDEOS_FOLDER, '*'))) #경로 뭉탱이를 리스트로
    names = [os.path.basename(folder) for folder in folders]  # only name
    for i, folder in enumerate(folders):
        name = names[i]
        videos = list(glob.iglob(os.path.join(folder, '*.*')))
        print('1', videos)
        save_folder = os.path.join(FACE_IMAGES_FOLDER, name)
        print(save_folder)
        os.makedirs(save_folder, exist_ok=True)
        for video in videos:
            print("videoname",video)
            extractor.get_frames(video, save_folder)
            crop_save_folder = os.path.join(FACE_IMAGES_FOLDER, name)
            #extractor.extract_faces(crop_save_folder)



if __name__ == '__main__':
    extractor = Make_datas()
    main(extractor)
