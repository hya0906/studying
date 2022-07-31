import os
import glob
import cv2
import numpy as np
import face_recognition
import pickle # save data as a file of pickle type

#recognizer - Eigenfaces,Fisherfaces,LBPH
DATA_PATH= "..\MakeData\Data\Face_images" # from root folder
encoding_file = 'face_model_15.pickle'
# model의 두 가지가 있다. hog와 cnn
# cnn은 높은 얼굴 인식 정확도를 보이지만 속도가 느리다는 단점(단, GPU환경은 빠르다)
# hog는 낮은 얼굴 인식 정확도를 보이지만 속도가 빠르다는 장점(cnn-gpu와 속도 비슷)
model_method = 'hog'

def main():
    Training_Data, Labels, Label_list = [], [], []
    folders = list(glob.iglob(os.path.join(DATA_PATH, '*')))
    print(folders)
    names = [os.path.basename(folder) for folder in folders]  # only names
    print(names)

    for name in names: # ['barack', 'IU', 'michelle', 'yuna_hong']
        folderpath = os.path.join(DATA_PATH, name)
        imgpath = list(glob.iglob(os.path.join(folderpath, '*.*')))
        #C:\Users\USER\Desktop\Facedlib\MakeData\Data\Face_images\IU\IU_avi_1.png
        if len(imgpath): # files exist
            print(folderpath, "-> files exist")
            for i, files in enumerate(imgpath):
                #if i == len(imgpath):#########
                #    break            #########
                #file_name = imgpath[i].split("\\")[-1]
                #print(file_name)

                #load image, turn BGR into RGB
                image = cv2.imread(files)
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # 입력 이미지에서 얼굴에 해당하는 box의 (x, y) 좌표 감지
                boxes = face_recognition.face_locations(rgb, model=model_method)

                # 얼굴 임베딩 계산
                encodings = face_recognition.face_encodings(rgb, boxes, model='small')

                # [tuple array][array([-0.09935994,...])]
                for encoding in encodings:
                    Training_Data.append(encoding)
                    Labels.append(name)
            print(name,"finish")

        else: # no files in the folder
            continue

    # save data as pickle file
    data = {"encodings": Training_Data, "names": Labels}
    f = open(encoding_file, "wb")
    f.write(pickle.dumps(data))
    f.close()

if __name__ == "__main__":
    main()