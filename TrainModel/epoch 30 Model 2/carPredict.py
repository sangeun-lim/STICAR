import torch
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import os
import matplotlib.pyplot as plt
import numpy as np

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.regnet_x_32gf(pretrained=False)
model.fc = nn.Linear(in_features=2520, out_features=54)
model.load_state_dict(torch.load("./epoch_30_54class.pt", map_location = torch.device('cpu')))
model = model.to(device)

data_transforms = {
    'result' : transforms.Compose([transforms.Resize((168,168)),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406],
                                                        [0.229, 0.224, 0.225])])
}

image_datasets = {'result': datasets.ImageFolder(root='./Folder', transform=data_transforms['result'])}
dataloaders = {'result': torch.utils.data.DataLoader(image_datasets['result'], batch_size=16)}
file_list = ['AudiA6', 'BMWX5', 'BenzCClass', 'BenzCLSClass', 'BenzGLCClass', 'BenzSClass', 'ChevroletCruze', 'ChevroletMalibu', 'ChevroletOrlando', 'ChevroletSpark', 'ChevroletTrax', 'FordExplorer', 'GenesisEQ900', 'GenesisG70', 'GenesisG80', 'GenesisG90', 'HyundaiAccent', 'HyundaiAvante', 'HyundaiGrandeur', 'HyundaiIoniq', 'HyundaiKona', 'HyundaiPalisade', 'HyundaiPorterII', 'HyundaiSantafe', 'HyundaiSonata', 'HyundaiStarex', 'HyundaiTucson', 'HyundaiVenue', 'Hyundaii30', 'KiaBongo3', 'KiaCarnival', 'KiaK3', 'KiaK5', 'KiaK7', 'KiaK9', 'KiaMohave', 'KiaMorning', 'KiaNiro', 'KiaRay', 'KiaSorento', 'KiaSportage', 'KiaStinger', 'KiaStonic', 'LandRoverDiscovery', 'LandRoverRangeRover', 'RenaultSamsungQM3', 'RenaultSamsungQM6', 'RenaultSamsungSM3', 'RenaultSamsungSM6', 'SsangYongKorandoC', 'SsangYongG4Rexton', 'SsangYongRextonSports', 'SsangYongTivoli', 'VolkswagenTiguan']
def test_visualize_model(model, num_images=1): # test Image  예측값을 보여주는 함수
    was_training = model.training
    model.eval() # 모델을 검증모드로
    images_so_far = 0
    # fig = plt.figure() # figure를 만들고 편집 할 수 있게 만들어주는 함수

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['result']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                # ax = plt.subplot(num_images//2, 2, images_so_far)
                print(file_list[preds[j]])
                # ax.axis('off')
                # ax.set_title('predict: {}'.format(class_names[preds[j]])) # 가장 높은확률의 이름 출력
                # imshow(inputs.cpu().data[j]) # 예측하려고 입력된 이미지 보여주기

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

test_visualize_model(model)
