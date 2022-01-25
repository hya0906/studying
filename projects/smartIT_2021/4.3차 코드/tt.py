from keras.models import load_model #save the model
import numpy as np
import winsound
saved_model = load_model('emotion_result.h5')
print(saved_model)
features = np.load('f.npy')  # 영상
print(features)


winsound.PlaySound('./audio/04_2.wav', winsound.SND_FILENAME)