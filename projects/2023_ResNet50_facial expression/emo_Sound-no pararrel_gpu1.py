'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import argparse
import numpy as np
from emo_resnet import *
from pytorch_cifar_master.utils import progress_bar
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
from torch.utils.data import random_split, Dataset
import matplotlib.pyplot as plt
import glob
import shutil
from matplotlib import pyplot as plt
import time
import datetime
import gc
import random
import librosa

def calculate_norm(dataset):
    mean_ = np.array([np.mean(x.numpy(), axis=(1, 2)) for x, _ in dataset])
    mean_r = mean_[:, 0].mean()
    mean_g = mean_[:, 1].mean()
    mean_b = mean_[:, 2].mean()
    std_ = np.array([np.std(x.numpy(), axis=(1, 2)) for x, _ in dataset])
    std_r = std_[:, 0].mean()
    std_g = std_[:, 1].mean()
    std_b = std_[:, 2].mean()
    return (mean_r, mean_g, mean_b), (std_r, std_g, std_b)

def matplotlib_imshow(img, one_channel=False):
    img = np.array(img.cpu())#########
    img = tf(img)
    tf = transforms.ToTensor()
    img = np.reshape(img, (-1, 32, 32, 3))

def images_to_probs(net, images):
    '''
    학습된 신경망과 이미지 목록으로부터 예측 결과 및 확률을 생성합니다
    '''
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.cpu().numpy())#####
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]

def plot_classes_preds(net, images, labels):
    '''
    학습된 신경망과 배치로부터 가져온 이미지 / 라벨을 사용하여 matplotlib
    Figure를 생성합니다. 이는 신경망의 예측 결과 / 확률과 함께 정답을 보여주며,
    예측 결과가 맞았는지 여부에 따라 색을 다르게 표시합니다. "images_to_probs"
    함수를 사용합니다.
    '''
    preds, probs = images_to_probs(net, images)
    # 배치에서 이미지를 가져와 예측 결과 / 정답과 함께 표시(plot)합니다
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]]),
                    color=("green" if preds[idx]==labels[idx].item() else "red"))
    return fig

# confusion matrix 그리는 함수
def plot_confusion_matrix(con_mat, labels, title='Confusion Matrix', cmap=plt.cm.get_cmap('Blues'), normalize=False):
    plt.imshow(con_mat, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    marks = np.arange(len(labels))
    nlabels = []
    for k in range(len(con_mat)):
        n = sum(con_mat[k])
        nlabel = '{0}(n={1})'.format(labels[k],n)
        nlabels.append(nlabel)
    plt.xticks(marks, labels)
    plt.yticks(marks, nlabels)

    thresh = con_mat.max() / 2.
    if normalize:
        for i, j in itertools.product(range(con_mat.shape[0]), range(con_mat.shape[1])):
            plt.text(j, i, '{0}%'.format(con_mat[i, j] * 100 / n), horizontalalignment="center", color="white" if con_mat[i, j] > thresh else "black")
    else:
        for i, j in itertools.product(range(con_mat.shape[0]), range(con_mat.shape[1])):
            plt.text(j, i, con_mat[i, j], horizontalalignment="center", color="white" if con_mat[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# Training
def train(epoch, trainloader):
    global net, device,criterion, optimizer
    net.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    for batch_idx, (input, target) in enumerate(trainloader):
        inputs, targets = input.to(device), target.to(device)
        optimizer.zero_grad() #gradients를 zero로 만들어주고 시작
        outputs = net(inputs)
        loss = criterion(outputs, targets) #cross entropy loss -> softmax + cross entropy
        loss.backward() #역전파
        optimizer.step() #경사하강법

        train_loss += loss.item() # estimate of the “epoch loss” during training.
        _, predicted = outputs.max(1)
        train_total += targets.size(0)
        train_correct += predicted.eq(targets).sum().item() #Computes element-wise equalityc
        progress_bar(batch_idx, len(trainloader), 'Loss: %.4f | Acc: %.4f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * train_correct / train_total, train_correct, train_total))

    train_loss = train_loss / len(trainloader)
    train_correct = 100. * train_correct/train_total
    return train_correct, train_total, train_loss

def valid(epoch, validloader):
    global net, device, criterion
    val_correct = 0.0
    val_loss = 0.0
    val_total = 0.0
    with torch.no_grad():
        net.eval()  # Optional when not using Model Specific layer
        for batch_idx, (inputs, targets) in enumerate(validloader):
            # Transfer Data to GPU if available
            if torch.cuda.is_available():
                inputs, targets = inputs.to(device), targets.to(device)

            # Forward Pass
            outputs = net(inputs)
            # Find the Loss
            loss = criterion(outputs, targets)
            # Calculate Loss
            val_loss += loss.item()

            _, predicted = outputs.max(1)
            val_total += targets.size(0)
            val_correct += predicted.eq(targets).sum().item()
            progress_bar(batch_idx, len(validloader), 'Val |Loss: %.4f | Acc: %.4f%% (%d/%d)'
                         % (val_loss / (batch_idx + 1), 100. * val_correct / val_total, val_correct, val_total))
    val_loss = val_loss / len(validloader)
    val_correct = val_correct * 100 / val_total
    return val_correct, val_total, val_loss

def test(epoch, testloader):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            progress_bar(batch_idx, len(testloader), 'Loss: %.4f | Acc: %.4f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc and acc > 75:
        print('Saving..')
        state = {
            'epoch': 200,
            'net': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'acc': acc,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        try:
            torch.save(state, f'./checkpoint/test_{round(acc,4)}_{batch_idx}.pth')
        except:
            torch.save(state, f'./checkpoint/test_{round(acc, 4)}_{batch_idx}_{random.randrange(1,21)}.pth')
        best_acc = acc
    test_loss = test_loss / len(testloader)
    correct = correct * 100 / total
    return correct, total, test_loss

def draw_confusion_matrix(testloader, classes, epoch, learning_rate):
    y_pred = []
    y_true = []
    plt.title(f"emotion-confusion matrix_Adam_{epoch}_{learning_rate}")
    # iterate over test data
    for inputs, labels in testloader:
        inputs = inputs.to(device)
        output = net(inputs)  # Feed Network

        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output)  # Save Prediction

        labels = labels.data.cpu().numpy()
        y_true.extend(labels)  # Save Truth
    print(y_pred, y_true)
    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    print("cf_matrix", cf_matrix)
    print("result", cf_matrix / np.sum(cf_matrix, axis=1)[:, None])
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=[i for i in classes],
                         columns=[i for i in classes])
    plt.figure(figsize=(12, 7))
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    sn.heatmap(df_cm, annot=True)
    plt.savefig(f"./emotion-confusion matrix_Adam_{epoch+1}_{learning_rate}.png")

def draw_graph(train_acc, valid_acc, test_acc, train_losses, valid_losses, test_losses, epoch):
    plt.figure(figsize=(12, 6))
    plt.figure(1)
    plt.subplot(1, 2, 1)
    plt.title("train/val accuracy")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(train_acc, label="train_acc", color="red")
    plt.plot(valid_acc, label="val_acc", color="blue")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title("train/val loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(train_losses, label="train_loss", color="red")
    plt.plot(valid_losses, label="val_loss", color="blue")
    plt.legend()
    plt.tight_layout()  ## <--마지막에 호출
    plt.savefig(f'./train_val_{epoch}_{random.randrange(1,21)}')
    # plt.show()

    plt.figure(figsize=(12, 6))  # 8:6
    plt.figure(2)
    plt.subplot(1, 2, 1)
    plt.title("test acc")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(test_acc, label="test_acc", color="green")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title("test loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(test_losses, label="test_loss", color="green")
    plt.legend()
    plt.tight_layout()  ## <--마지막에 호출
    plt.savefig(f'./test_{epoch}_{random.randrange(1,21)}')
    # plt.show()


MFCC_X = 20
MFCC_Y = 44


def _path(loc, directory, filename, redundant, format_):
    delimiter = '\\'
    path = (loc + delimiter + directory + delimiter +
            filename + redundant + format_)
    return path


def _normalize(mfcc):
    return (mfcc - np.min(mfcc)) / (np.max(mfcc) - np.min(mfcc))


def _fit_size(arr, size):
    zeros = np.zeros((len(arr), size - len(arr[0])))
    arr = np.append(arr, zeros, axis=1)
    arr.astype('float32')
    return arr


def _mfcc(path):
    y, sr = librosa.load(path)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    mfcc = _fit_size(mfcc, MFCC_Y)
    return _normalize(mfcc)

class SoundDataset(Dataset):
    def parse_audio_files(parent_dir, sub_dirs, file_ext="*.wav"):
        features, labels = np.empty((0, 193)), np.empty(0)
        for label, sub_dir in enumerate(sub_dirs):
            for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
                try:
                    mfccs, chroma, mel, contrast, tonnetz = extract_feature(fn)
                except Exception as e:
                    print("Error encountered while parsing file: ", fn)
                    continue
                ext_features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
                features = np.vstack([features, ext_features])
                labels = np.append(labels, fn.split('/')[2].split('-')[2])
        return np.array(features), np.array(labels, dtype=np.int)

    def trim_audio(self, audio_file, save_file, length):
        sr = 96000
        sec = int(length) // 5  # 3
        a = 0
        audio, sr = librosa.load(audio_file, sr=sr)
        duration = librosa.get_duration(audio, sr=sr)
        print(duration)
        '''
        while (sr * (sec * (a + 1)) < len(audio)):
            ny = audio[sr * (0 + sec * a):sr * (sec * (a + 1))]
            librosa.output.write_wav(save_file + str(self.i + 1) + '.wav', ny, sr)
            print(save_file + str(self.i + 1) + ".wav완료")
            self.i += 1
            a += 1
        '''

    def __init__(self, train=True):
        SOUND_PATH = "C:\\Users\\711_2\\Desktop\\Yuna_Hong\\facial_expression\\emo_project_test\\Audio_Speech"
        label_list = {"01" : "neutral", "02" : "calm", "03" : "happy", "04" : "sad", "05" : "angry",
                      "06" : "fearful", "07" : "disgust", "08" : "surprised"}
        Kfold={"0": [2, 5, 14, 15, 16], "1": [3, 6, 7, 13, 18], "2": [10, 11, 12, 19, 20], "3": [8, 17, 21, 23, 24], "4": [1, 4, 9, 22]}


        self.train = train
        num_data = 100
        num_train = 90
        num_test = 10

        folders = list(glob.iglob(os.path.join(SOUND_PATH, '*')))  # 경로 뭉탱이를 리스트로
        names = [os.path.basename(folder) for folder in folders]  # only name
        for i, folder in enumerate(folders):
            name = names[i]
            videos = list(glob.iglob(os.path.join(folder, '*.*')))
            print('1', videos)
            save_folder = os.path.join(FACE_IMAGES_FOLDER, name)
            print(save_folder)
            os.makedirs(save_folder, exist_ok=True)

        '''
        if self.train == True:
            self.train_data = []
            self.train_label = []
        
            print("\n\n==== Train Data:")
            for item in label_list:
                print("===", end="")
                for i in range(1, num_train + 1):
                    path = _path(loc='.\\dataset',
                                 directory=self.label_word[item], filename=self.label_word[item],
                                 redundant=' (' + str(i) + ')', format_='.wav')
                    mfcc = _mfcc(path)
                    self.train_data.append(mfcc)
                    self.train_label.append(item)

            self.train_label = np.array(self.train_label)
            self.train_data = np.concatenate(self.train_data)
            self.train_data = self.train_data.reshape(num_train * 4, MFCC_X, MFCC_Y)  # 3600개에 (20,44) mfcc
            print("=== Dataset Download Complete !!")

        else:
            self.test_data = []
            self.test_label = []

            print("\n\n=== Test Data:")
            for item in label_list:
                print("===", end="")
                for i in range(num_train + 1, num_data + 1):
                    path = _path(loc='.\\dataset',
                                 directory=self.label_word[item], filename=self.label_word[item],
                                 redundant=' (' + str(i) + ')', format_='.wav')
                    mfcc = _mfcc(path)
                    self.test_data.append(mfcc)
                    self.test_label.append(item)

            self.test_label = np.array(self.test_label)
            self.test_data = np.concatenate(self.test_data)
            self.test_data = self.test_data.reshape(num_test * 4, MFCC_X, MFCC_Y)  # 3600개에 (20,44) mfcc
            print("=== Dataset Download Complete !!")
       '''

    def __getitem__(self, index):
        if self.train:
            return self.train_data[index], self.train_label[index]
        else:
            return self.test_data[index], self.test_label[index]

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

def get_train_loader(batch_size, num_worker):
    trainset = SoundDataset(train=True)
    testset = SoundDataset(train=False)

    train_size = int(0.9 * len(trainset))
    valid_size = len(trainset) - train_size

    trainset, _ = torch.utils.data.random_split(trainset, [train_size, valid_size])
    # iterator과 next로 데이터 비교해서 validset새로 만들기
    _, validset = torch.utils.data.random_split(validset, [train_size, valid_size])

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size= batch_size, shuffle=True, num_workers=num_worker)
    validloader = torch.utils.data.DataLoader(
        validset, batch_size=batch_size, shuffle=True, num_workers=num_worker)
    return trainloader, validloader

def resume_from(file_name):
    # 저장했던 중간 모델 정보를 읽습니다.
    model_data = torch.load(file_name)

    model = ResNet50(model_data['input_dim'])
    # 저장했던 모델의 값들로 치환합니다.
    model.load_state_dict(model_data['model_state_dict'])

    optimizer = optim.Adam(model.parameters())
    # optimizer도 중간에 저장했던 값들로 치환합니다.

    optimizer.load_state_dict(model_data['optimizer_state_dict'])

    # 지금 시작할 epoch은 기존 epoch + 1 즉 다음 epoch입니다.
    start_epoch = model_data['epoch'] + 1
    return start_epoch, optimizer, model

def main_worker():
    global net, trainloader,testloader, device, best_acc, start_epoch, epochs, criterion, optimizer

    GPU_NUM = 0  # 원하는 GPU 번호 입력
    device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)  # change allocation of current GPU
    print('Current cuda device ', torch.cuda.current_device(),'\n')  # check

    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    batch_size = 16
    num_worker = 8
    epochs = 80
    learning_rate = 0.001

    net = ResNet50()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, betas=(0.9, 0.999))

    start = input("중간에서부터 다시 시작하시겠습니까?(y/n)")
    if start == 'y':
        file_name = input("모델 파일 이름을 적어주세요")
        start_epoch, optimizer, net = resume_from(file_name)
        print("이어서 학습")
    else:
        print("처음부터 학습")

    net = net.to(device)

    # Data
    print('==> Preparing data..')
    trainloader, validloader = get_train_loader(batch_size, num_worker)
    transform_test = transforms.Compose([
        transforms.Resize(112),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    testset = torchvision.datasets.ImageFolder(root="C:/Users/711_2/Desktop/Yuna_Hong/facial_expression/aligned/test",
                                               transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False, num_workers=4)
    print("train",len(trainloader))
    print("valid",len(validloader))
    print("test",len(testloader))

    # mean, std = calculate_norm(trainset)
    # print(mean, std)

    classes = ('surprised', 'fearful', 'disgusted', 'happy', 'sad', 'angry', 'neutral')

    # Model
    print('==> Building model..')

    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(net.parameters(), lr=args.lr,
    #                      momentum=0.9, weight_decay=5e-4)
    #optimizer위로 옮김
    #optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    # learning rate가 단순히 감소하기 보다는 진동하면서 최적점을 찾아가는 방식
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    start = time.time()
    train_acc, valid_acc, test_acc = [], [], []
    train_losses, valid_losses, test_losses = [], [], []
    for epoch in range(start_epoch, epochs):
        print('\nEpoch: %d' % epoch)
        train_correct, train_total, train_loss = train(epoch, trainloader)
        train_acc.append(train_correct)
        train_losses.append(train_loss)
        val_correct, val_total, val_loss = valid(epoch, validloader)
        valid_acc.append(val_correct)
        valid_losses.append(val_loss)
        test_correct, test_total, test_loss = test(epoch, testloader)
        test_acc.append(test_correct)
        test_losses.append(test_loss)
        scheduler.step()  # Decays the learning rate of each parameter group by gamma every step_size epochs
        torch.save({
            'epoch': epoch,
            'optimizer_state_dict': optimizer.state_dict(),
            'model_state_dict': net.state_dict(),
            'input_dim': 112
        }, os.path.join("emo_project_test/epoch_weight/112_Adam_80_0.001_bs16,16_t,v,t sepdata", 'epoch-{}.pth'.format(epoch)))
    end = time.time()
    sec = end - start
    times = str(datetime.timedelta(seconds=sec)).split(".")
    print(times[0])

    draw_confusion_matrix(testloader, classes, epoch, learning_rate)
    draw_graph(train_acc, valid_acc, test_acc, train_losses, valid_losses, test_losses, epoch)



if __name__ == "__main__":
    gc.collect()  # cuda 캐시 비워줌
    torch.cuda.empty_cache()
    #main()
    main_worker()
