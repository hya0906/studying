from PIL import Image
import os

raw_path = "C:\\Users\\711_2\\Downloads\\sum_data\\train_" # 원본 이미지 경로
token_list = os.listdir(raw_path) # 원본 이미지 경로 내 폴더들 list
data_path = "C:\\Users\\711_2\\Downloads\\sum_data\\train_"  # 저장할 이미지 경로

# resize
for cls in token_list:
    # 원본 이미지 경로와 저장할 경로 이미지 지정
    image_path = raw_path+ '\\' + cls + '\\'
    save_path = data_path+ '\\' + cls + '\\'
    print(image_path, save_path)

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # 원본 이미지 경로의 모든 이미지 list 지정
    data_list = os.listdir(image_path)

    # 모든 이미지 resize 후 저장하기
    for name in data_list[1500:3000]:
        im = Image.open(image_path + name)# 이미지 열기
        im = im.resize((56, 56))# 이미지 resize
        im = im.convert('RGB')# 이미지 JPG로 저장
        im.save(save_path + name)
    print('end ::: ' + cls)
