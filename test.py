import os
import json
from tqdm import tqdm
import sys
import torch
import SimpleITK as sitk
from torchvision import transforms
import matplotlib.pyplot as plt
from utils import Logger
from net.resnet import resnet34
from net.alexnet import AlexNet
from net.googlenet import GoogLeNet
from eval import cal_mAP


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.ToPILImage(),
         transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize((0.5), (0.5))])

    # load image
    img_dir = "data/test_set"
    assert os.path.exists(img_dir), "file: '{}' dose not exist.".format(img_dir)
    
     # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)
    
    logfile = './results/google_log_test.txt'
    sys.stdout = Logger(logfile)
    
    # create model
    # model = resnet34(num_classes=6).to(device)
    # model = AlexNet(num_classes=6).to(device)
    model = GoogLeNet(num_classes=6, aux_logits=False, init_weights=True).to(device)


    # load model weights
    weights_path = "weights/google_best_model.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    
    img_path_list = []
    for classes in os.listdir(img_dir):
        class_dir = os.path.join(img_dir, classes)
        for img in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img)
            img_path_list.append(img_path)
    
    all_num = 0
    right_num = 0
    target = {'0':[], '1':[],'2':[],'3':[],'4':[],'5':[]}
    pred = {'0':[], '1':[],'2':[],'3':[],'4':[],'5':[]}
    
    
    
    img_loader = tqdm(img_path_list)
    for img_path in img_loader:
        all_num += 1
        original_img = sitk.ReadImage(img_path)
        img_array = sitk.GetArrayFromImage(original_img)
        img = torch.from_numpy(img_array)
        
        if len(img.shape) == 4:
            img = img[:,:,:,0]
            
        img = data_transform(img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        truth_class = os.path.dirname(img_path).split('/')[-1]
        
        img_loader.desc = '[Predicting]'

        # prediction
        model.eval()
        with torch.no_grad():
            # predict class
            output = torch.squeeze(model(img.to(device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()

        print_res = "predict_class: {}   prob: {:.3}    truth_class: {}".format(class_indict[str(predict_cla)],
                                                    predict[predict_cla].numpy(), truth_class)
        pred['0'].append(predict[0].numpy())
        pred['1'].append(predict[1].numpy())
        pred['2'].append(predict[2].numpy())
        pred['3'].append(predict[3].numpy())
        pred['4'].append(predict[4].numpy())
        pred['5'].append(predict[5].numpy())
            
        if truth_class == '123':
            target['0'].append(1)
        else:
            target['0'].append(0)
        if truth_class == '1234':
            target['1'].append(1)
        else:
            target['1'].append(0)
        if truth_class == '4':
            target['2'].append(1)
        else:
            target['2'].append(0)
        if truth_class == '5678':
            target['3'].append(1)
        else:
            target['3'].append(0)
        if truth_class == '58':
            target['4'].append(1)
        else:
            target['4'].append(0)
        if truth_class == '67':
            target['5'].append(1)
        else:
            target['5'].append(0)
            
        
        # print(print_res)
        if class_indict[str(predict_cla)] == truth_class:
            right_num += 1
    # for i in range(len(predict)):
    #     print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
    #                                               predict[i].numpy()))
    mAP = cal_mAP(target, pred)

    print('[Accuracy]: {}'.format(right_num/all_num))
    print('[mAP]: {}'.format(mAP))


if __name__ == '__main__':
    main()
