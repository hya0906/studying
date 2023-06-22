1. code 폴더의 파일 설명
 - pretrained.py : 모델 학습
 - resize_img : 데이터 이미지 resize(저해상도로 만들기)
 - test_emo : 카메라에 실시간으로 감정인식 하기★
 - utils : progress bar 사용하기 위한 파일
 - Columbia_test: 조석주님 모델에 emotion predict부분 추가해 본 것.★


2.모델/사용 데이터셋 설명
 -Model: ResNet50 +pretrained weight(ResNet50_Weights.IMAGENET1K_V2 (80.858%))
   ->pretrain한 모델의 weightsms ImageNet 버전1과 버전 2가 있는데 파이토치 문서에 따르면 버전2가 더 정확도가 높아 이를 지정하였다.
   ->모델의 마지막 부분만 (2048,1000)을 (2048,7)로 변경

 -Dataset: Raf-DB_resize
   ->Raf-DB 이미지 데이터를 일부분 (56,56)으로 resize시켜서 학습할때 112x112로 키워서 저해상도에서도 잘 인식할 수 있도록 하였다.
   ->train/val data 비율은 9:1
   ->classes = {0:'surprised', 1:'fearful', 2:'disgusted', 3:'happy', 4:'sad', 5:'angry', 6:'neutral'}

 -Augmentation: 감정인식을 위한 augmentation인 resize+crop으로 zoom in하였고 flip과 실제 상황에서 잘 인식이 되게 하기 위해 
   조도조절, 해상도 조절을 하였다.

 -Optimizer: Adam/ learning rate: 0.00025/ train/test batch size: 16/ num worker: 4(gpu 2080ti 1개 사용)/ scheduler: CosineAnnealingLR(optimizer, T_max=200) 사용

 -모델 선정 방법: test accuracy, confusion matrix, F1 score을 사용하였다. Raf-DB는 클래스가 불균형한 모델이기 때문에 test acc과 F1 score가 균형있게 나오고 
   confusion matrix도 상대적으로 잘 나온 모델을 선정하였다.

 -모델 저장: torch.save({
                               'epoch': epoch,
                               'optimizer_state_dict': optimizer.state_dict(),
                               'model_state_dict': net.state_dict()
                             }, os.path.join(f".\\weight\\{folder}", 'epoch-{}.pt'.format(epoch)))

 -문제점: 카메라와 거의 얼굴이 정면이어야 인식이 된다. 그리고 표정을 애매하게 지으면 안되고 정확하게 지어야 한다.


3.test_emo.py
   ->실시간으로 예측을 할 때 예측값이 일정하지 않기 때문에 fps의 1/5로 지정하였음.(현재 30fps 카메라 사용중)
   ->★★★감정인식을 위한 함수는 test_img. roi_color을 얻기 위해 조석주님의 Columbia_test.py의 detect하는 부분을 받아야 함.★★★
   ->Columbia_test의 visualization함수에 vidTracks 변수로 얼굴부분 rectangle 그림.
  

★★★★★★★★★중요
3.1 test_emo.py 쓰는 함수(조석주님 모델로 얼굴detect하고 그 영역으로 얼굴감정 predict)
   ->합칠때는 haarcascade필요없음
   ->★★★load_model_and_data(모델,transform)와 test_img(예측)함수 필요.★★★
   ->1/5 프레임씩 출력코드(경우에 따라 추가/삭제)
            if len(emotion) == 6: #1/5프레임당 가장 많이 나온 값 선정-정확도를 위해(변경가능)
                count = Counter(emotion)
                c = count.most_common(n=1)
                cls = classes[c[0][0]]#클래스 이름만
                print(cls)
                emotion.clear()
  
4.Columbia_test.py
 - 조석주님 모델(LightASD의 Columbia_test.py)에 emotion predict부분 추가해 본 것.★
 - 라이브러리 import 때문에 실행 못해봄
예시(수정한 부분은 #&추가)
else:
	# Visualization, save the result as the new video
	net, transform, face_cascade = load_model_and_data() #&
	roi_color = visualization(vidTracks, scores, args) #&
	conf, class_ = test_img(roi_color)  # 정확도, 클래스 인덱스 #&
	print(class_) #&
★★★★★★★★★중요