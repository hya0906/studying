ResNet50으로 facial expression 학습  
Raf-DB사용  
Single GPU / Multi GPU  

## 연구실 컴퓨터 GEFORCE RTX 2080ti
-https://webnautes.tistory.com/1454#google_vignette
-https://blog.nerdfactory.ai/2021/04/30/Setting-up-CUDA-11.1.html
- cuda: 11.2  https://developer.nvidia.com/cuda-11.2.2-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exelocal
- cudnn: 8.1.1 (또는 8.1.0)  https://developer.nvidia.com/rdp/cudnn-archive
- NVIDIA Studio 드라이버: 528.49  https://www.nvidia.co.kr/download/driverResults.aspx/199792/kr
- tensorflow-gpu: 2.9.0  https://www.tensorflow.org/install/source_windows?hl=ko#gpu

# 참고 사이트
## 1. Multi GPU 
- https://medium.com/daangn/pytorch-multi-gpu-%ED%95%99%EC%8A%B5-%EC%A0%9C%EB%8C%80%EB%A1%9C-%ED%95%98%EA%B8%B0-27270617936b  
- https://velog.io/@khs0415p/Pytorch-Multi-GPU-%ED%95%99%EC%8A%B5  
- emo-main_no parrarel.py / emo-main_parrel.py multi GPU/ singualr GPU

## 2. emo_data.py  
- 사진데이터를 클래스로 폴더로 분류. 레이블은 메모장으로 따로 존재.  
- data : aligned.vol1/vol2.egg, label : list_patition_label.txt
  
## 3. 소리데이터  
![다운로드](https://user-images.githubusercontent.com/59861622/235306897-d06588e0-147b-4f66-8550-26eb5361d0ce.png)  
- data : https://zenodo.org/record/1188976#.ZE0iZnZBxPZ
- custom dataset만들기: https://m.blog.naver.com/sooftware/221646956569

