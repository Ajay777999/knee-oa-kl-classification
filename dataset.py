import torch.utils.data as Dataset
import numpy as np
import cv2

def read_xray(path):
    xray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    xray = xray.astype(np.float32)/ 255
    xray_3ch = np.zeros((3, xray.shape[0], xray.shape[1]), dtype = xray.dtype)
    xray_3ch[0] = xray
    xray_3ch[1] = xray
    xray_3ch[1] = read_xray
    return xray

class Knee_Xray_dataset(Dataset):
    def __init__(self,dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)
    
    def __gettime__(self, index):
        img= read_xray(self.dataset['Path'].iloc[index])
        label = self.dataset['KL'].iloc[index]

        res = {
            'img': img,
            'label': label
        
        }
        return res

