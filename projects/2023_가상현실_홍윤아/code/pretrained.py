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
from utils import progress_bar
from sklearn.metrics import confusion_matrix, f1_score
import seaborn as sn
import pandas as pd
from torch.utils.data import random_split
from matplotlib import pyplot as plt
import time
import datetime
import gc
from tqdm import tqdm

folder = "no_freeze_v2_0.00015+sum"

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
    for batch_idx, (input, target) in enumerate(tqdm(trainloader, unit= "batch")):
        inputs, targets = input.cuda(), target.cuda()
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
    global net, device, criterion, optimizer
    val_correct = 0.0
    val_loss = 0.0
    val_total = 0.0
    with torch.no_grad():
        net.eval()  # Optional when not using Model Specific layer
        for batch_idx, (inputs, targets) in enumerate(tqdm(validloader, unit="batch")):
            # Transfer Data to GPU if available
            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()

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
    global best_acc, optimizer, net
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(testloader, unit = "batch")):
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
        torch.save(state, f'.\\checkpoint\\{folder}\\test_epoch{epoch}_{round(acc,4)}_{batch_idx}.pth')
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
        inputs = inputs.cuda()
        output = net(inputs)  # Feed Network

        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output)  # Save Prediction

        labels = labels.data.cpu().numpy()
        y_true.extend(labels)  # Save Truth
    score = f1_score(y_true, y_pred, pos_label=7, average = 'macro')
    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=[i for i in classes],
                         columns=[i for i in classes])
    plt.figure(figsize=(12, 7))
    plt.title(f"F1 score: {score}")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    sn.heatmap(df_cm, annot=True)
    plt.savefig(f".\\graphs\\{folder}\\emotion-confusion matrix_Adam_{epoch+1}_{learning_rate}.png")


def get_train_loader(batch_size, num_worker):
    transform_train = transforms.Compose([
        transforms.Resize(116),
        transforms.RandomCrop(112),  # image size 100 X 100
        transforms.ColorJitter(brightness=0.5),
        transforms.RandomHorizontalFlip(),  # 50%의 확률로 뒤집음
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # 각 channel 에 맞춰서 normalize
    ])
    trainset = torchvision.datasets.ImageFolder(root="C:\\Users\\711_2\\Downloads\\sum_data\\train_",
                                                transform=transform_train)
    print("hello",len(trainset))

    train_size = int(0.9 * len(trainset))
    valid_size = len(trainset) - train_size


    train_set, valid_set = torch.utils.data.random_split(trainset, [train_size, valid_size])

    trainloader = torch.utils.data.DataLoader(
        train_set, batch_size= batch_size, shuffle=True, num_workers=num_worker)
    validloader = torch.utils.data.DataLoader(
        valid_set, batch_size=batch_size, shuffle=True, num_workers=num_worker)
    return trainloader, validloader

def main():
    global net, trainloader,testloader, device, best_acc, start_epoch, epochs, criterion, optimizer
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    batch_size = 32
    num_worker = 8
    epochs = 50
    learning_rate = 0.00015

    net = torchvision.models.resnet50(weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    net = net.cuda()

    # 모델 구조 수정
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, 7)
    net = net.to(device)
    print(net.fc)

    # Data
    print('==> Preparing data..')
    trainloader, validloader = get_train_loader(16, 4)
    transform_test = transforms.Compose([
        transforms.Resize(112),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), #imagenet
    ])

    testset = torchvision.datasets.ImageFolder(root="C:\\Users\\711_2\\Downloads\\sum_data\\test",
                                               transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False, num_workers=4)
    print("train",len(trainloader))
    print("valid",len(validloader))
    print("test",len(testloader))

    classes = ('surprised', 'fearful', 'disgusted', 'happy', 'sad', 'angry', 'neutral')

    # Model
    print('==> Building model..')

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, betas=(0.9, 0.999))
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
        torch.save(net, f'.\\models\\{folder}\\model-{epoch}.pt')
        torch.save({
            'epoch': epoch,
            'optimizer_state_dict': optimizer.state_dict(),
            'model_state_dict': net.state_dict()
        }, os.path.join(f".\\weight\\{folder}", 'epoch-{}.pt'.format(epoch)))
        draw_confusion_matrix(testloader, classes, epoch, learning_rate)
        if epoch % 4 == 0:
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
            plt.savefig(f'.\\graphs\\{folder}\\train_val_{epoch}')
            plt.clf()

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
            plt.savefig(f'.\\graphs\\{folder}\\test_{epoch}')
            plt.clf()

    end = time.time()
    sec = end - start
    times = str(datetime.timedelta(seconds=sec)).split(".")
    print(times[0])


if __name__ == "__main__":
    gc.collect()  # cuda 캐시 비워줌
    torch.cuda.empty_cache()
    main()

