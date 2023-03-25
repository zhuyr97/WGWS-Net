import os,random,glob,math
import torch.nn as nn
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import Dataset,DataLoader

def read_txt(txt_name = 'RealHaze.txt',sample_num=5000):
    path_in = []
    path_gt = []
    paths =[]
    with open(txt_name, 'r') as f:  # RealSnow
        for line in f:
            # print(line.strip('\n'))
            paths.append(line.strip('\n'))
    if sample_num > len(paths):
        sample_num = len(paths)
    paths_random = random.sample(paths, sample_num)
    for path in paths_random:
        path_in.append(path.strip('\n').split(' ')[0])
        path_gt.append(path.strip('\n').split(' ')[1])
    return path_in,path_gt

class my_dataset(Dataset):
    def __init__(self, rootA, rootA_txt, rootB, rootB_txt,rootC, rootC_txt, crop_size =256,
                 fix_sample_A = 500,fix_sample_B =500 ,fix_sample_C =500,regular_aug =False):
        super(my_dataset,self).__init__()

        self.regular_aug = regular_aug
        #in_imgs
        self.fix_sample_A = fix_sample_A

        in_files_A, gt_files_A = read_txt(rootA_txt, sample_num=self.fix_sample_A)  #os.listdir(rootA_in)
        self.imgs_in_A = [rootA + k for k in in_files_A]#os.path.join(rootA_in, k)
        self.imgs_gt_A = [rootA + k for k in gt_files_A]#gt_imgs  os.path.join(rootA_label

        len_imgs_in_A = len(self.imgs_in_A)
        self.length = len_imgs_in_A
        self.r_l_rate = 1 #split_rate[1] // split_rate[0]  #(1, 1)时 r_l_rate = 1
        self.r_l_rate1 = 1  #split_rate[2] // split_rate[0]  #(1, 1)时 r_l_rate = 1


        #in_files_B = os.listdir(rootB_in)
        in_files_B, gt_files_B = read_txt(rootB_txt, sample_num = fix_sample_B)
        self.imgs_in_B = [rootB + k for k in in_files_B] #[os.path.join(rootB_in, k) for k in in_files_B]
        self.imgs_gt_B = [rootB + k for k in gt_files_B] # gt_imgs  name keep same

        len_imgs_in_B_ori = len(self.imgs_in_B )# 在 B 中 相较于imgs_A所缺的数目 补齐img_B图片
        self.imgs_in_B = self.imgs_in_B * (self.r_l_rate + math.ceil(len_imgs_in_A / len_imgs_in_B_ori))  # 扩展
        self.imgs_in_B = self.imgs_in_B[0: self.r_l_rate * len_imgs_in_A]
        self.imgs_gt_B = self.imgs_gt_B * (self.r_l_rate + math.ceil(len_imgs_in_A / len_imgs_in_B_ori))
        self.imgs_gt_B = self.imgs_gt_B[0: self.r_l_rate * len_imgs_in_A]

        in_files_C, gt_files_C = read_txt(rootC_txt, sample_num=fix_sample_C)
       # in_files_C = os.listdir(rootC_in)

        self.imgs_in_C = [rootC + k for k in in_files_C] #[os.path.join(rootC_in, k) for k in in_files_C]
        self.imgs_gt_C = [rootC + k for k in gt_files_C] #[os.path.join(rootC_label, k) for k in in_files_C]  # gt_imgs  name keep same

        len_imgs_in_C_ori = len(self.imgs_in_C)  # 在 C 中  相较于imgs_A所缺的数目 补齐img_C图片
        self.imgs_in_C = self.imgs_in_C * (self.r_l_rate1 + math.ceil(len_imgs_in_A / len_imgs_in_C_ori))  # 扩展
        self.imgs_in_C = self.imgs_in_C[0: self.r_l_rate1 * len_imgs_in_A]
        self.imgs_gt_C = self.imgs_gt_C * (self.r_l_rate1 + math.ceil(len_imgs_in_A / len_imgs_in_C_ori))
        self.imgs_gt_C = self.imgs_gt_C[0: self.r_l_rate1 * len_imgs_in_A]

        self.crop_size = crop_size
    def __getitem__(self, index):
        data_IN_A, data_GT_A, img_name_A, gt_name_A = self.read_imgs_pair(self.imgs_in_A[index], self.imgs_gt_A[index], self.train_transform, self.crop_size)
        data_IN_B, data_GT_B, img_name_B, gt_name_B = self.read_imgs_pair(self.imgs_in_B[index], self.imgs_gt_B[index], self.train_transform, self.crop_size)
        data_IN_C, data_GT_C, img_name_C, gt_name_C = self.read_imgs_pair(self.imgs_in_C[index], self.imgs_gt_C[index], self.train_transform, self.crop_size)

        data_A = [data_IN_A, data_GT_A, img_name_A, gt_name_A]
        data_B = [data_IN_B, data_GT_B, img_name_B, gt_name_B]
        data_C = [data_IN_C, data_GT_C, img_name_C, gt_name_C]
        return data_A, data_B, data_C

    def read_imgs_pair(self,in_path, gt_path, transform, crop_size):
        in_img_path_A = in_path  #
        img_name_A = in_img_path_A.split('/')[-1]

        in_img_A = np.array(Image.open(in_img_path_A))
        gt_img_path_A = gt_path  # self.imgs_gt_A[index]

        gt_name_A = gt_img_path_A.split('/')[-1]
        gt_img_A = np.array(Image.open(gt_img_path_A))
        data_IN_A, data_GT_A = transform(in_img_A, gt_img_A, crop_size)

        return data_IN_A, data_GT_A, img_name_A, gt_name_A
    def augment_img(self, img, mode=0):
        """图片随机旋转"""
        if mode == 0:
            return img
        elif mode == 1:
            return np.flipud(np.rot90(img))
        elif mode == 2:
            return np.flipud(img)
        elif mode == 3:
            return np.rot90(img, k=3)
        elif mode == 4:
            return np.flipud(np.rot90(img, k=2))
        elif mode == 5:
            return np.rot90(img)
        elif mode == 6:
            return np.rot90(img, k=2)
        elif mode == 7:
            return np.flipud(np.flipud(np.rot90(img, k=3)))

    def train_transform(self, img, label,patch_size=256):
        """对图片和标签做一些数值处理"""
        ih, iw,_ = img.shape

        patch_size = patch_size
        ix = random.randrange(0, max(0, iw - patch_size))
        iy = random.randrange(0, max(0, ih - patch_size))
        img = img[iy:iy + patch_size, ix: ix + patch_size]
        label = label[iy:iy + patch_size, ix: ix + patch_size]

        if self.regular_aug:
            mode = random.randint(0, 7)
            img = self.augment_img(img, mode=mode)
            label = self.augment_img(label, mode=mode)
            img = img.copy()
            label = label.copy()

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        img = transform(img)
        label = transform(label)

        return img, label

    def __len__(self):
        return len(self.imgs_in_A)

class my_dataset_eval(Dataset):
    def __init__(self,root_in,root_label,transform =None,fix_sample=100):
        super(my_dataset_eval,self).__init__()
        #in_imgs

        self.fix_sample = fix_sample
        in_files = os.listdir(root_in)
        if self.fix_sample > len(in_files):
            self.fix_sample = len(in_files)
        in_files = random.sample(in_files, self.fix_sample)
        self.imgs_in = [os.path.join(root_in, k) for k in in_files]
        #gt_imgs
        #gt_files = os.listdir(root_label)
        self.imgs_gt = [os.path.join(root_label, k) for k in in_files]

        self.transform = transform
    def __getitem__(self, index):
        in_img_path = self.imgs_in[index]
        img_name =in_img_path.split('/')[-1]

        in_img = Image.open(in_img_path)
        gt_img_path = self.imgs_gt[index]
        gt_img = Image.open(gt_img_path)
        trans_eval = transforms.Compose(
            [
                transforms.ToTensor()
            ])

        data_IN = trans_eval(in_img)
        data_GT = self.transform(gt_img)

        _, h, w = data_GT.shape
        if (h % 32 != 0) or (w % 32 != 0):
            data_GT = transforms.Resize(((h // 32) * 32, (w // 32) * 32))(data_GT)
            data_IN = transforms.Resize(((h // 32) * 32, (w // 32) * 32))(data_IN)

        return data_IN, data_GT, img_name

    def __len__(self):
        return len(self.imgs_in)

class my_dataset_eval_realH(Dataset):
    def __init__(self, root_in, root_label, transform=None, fix_sample=100):
        super(my_dataset_eval_realH, self).__init__()
        # in_imgs
        self.fix_sample = fix_sample

        in_files = os.listdir(root_in)
        if len(in_files) < self.fix_sample:
            in_files = in_files
        else:
            in_files = random.sample(in_files, self.fix_sample)
        self.imgs_in = [os.path.join(root_in, k) for k in in_files]
        # gt_imgs
        gt_files = os.listdir(root_label)
        self.imgs_gt = [os.path.join(root_label, k) for k in gt_files]
        print("testing dataset different???  ===================================")
        print('dataname of input & GT:', os.listdir(root_in) == os.listdir(root_label))

    def __getitem__(self, index):
        in_img_path = self.imgs_in[index]
        img_name = in_img_path.split('/')[-1]

        in_img = Image.open(in_img_path)
        gt_img_path = self.imgs_gt[index]
        gt_img = Image.open(gt_img_path)

        width_ori, height_ori = in_img.size
        in_img = in_img.resize((int(width_ori * 0.5), int(height_ori * 0.5)), Image.BILINEAR)
        gt_img = gt_img.resize((int(width_ori * 0.5), int(height_ori * 0.5)), Image.BILINEAR)# gt_img #

        trans_eval = transforms.Compose(
            [
                transforms.ToTensor()
            ])

        data_IN = trans_eval(in_img)
        data_GT = trans_eval(gt_img)

        _, h, w = data_GT.shape
        # print('before',data_IN.shape,data_GT.shape)
        if (h % 32 != 0) or (w % 32 != 0):
            data_GT = transforms.Resize(((h // 32) * 32, (w // 32) * 32))(data_GT)
            data_IN = transforms.Resize(((h // 32) * 32, (w // 32) * 32))(data_IN)
        # print('after',data_IN.shape,data_GT.shape)
        # print()
        return data_IN, data_GT, img_name

    def __len__(self):
        return len(self.imgs_in)



class my_dataset_eval_Transweather(Dataset):
    def __init__(self,root_in,root_label,transform =None,fix_sample=100):
        super(my_dataset_eval_Transweather,self).__init__()
        #in_imgs

        self.fix_sample = fix_sample
        in_files = os.listdir(root_in)
        if self.fix_sample > len(in_files):
            self.fix_sample = len(in_files)
        in_files = random.sample(in_files, self.fix_sample)
        self.imgs_in = [os.path.join(root_in, k) for k in in_files]
        #gt_imgs
        #gt_files = os.listdir(root_label)
        self.imgs_gt = [os.path.join(root_label, k) for k in in_files]

        self.transform = transform
    def __getitem__(self, index):
        in_img_path = self.imgs_in[index]
        img_name =in_img_path.split('/')[-1]

        in_img = Image.open(in_img_path)
        gt_img_path = self.imgs_gt[index]
        gt_img = Image.open(gt_img_path)
        trans_eval = transforms.Compose(
            [
                transforms.ToTensor()
            ])

        data_IN = trans_eval(in_img)
        data_GT = self.transform(gt_img)

        _, h, w = data_GT.shape
        if (h % 32 != 0) or (w % 32 != 0):
            data_GT = transforms.Resize(((h // 32) * 32, (w // 32) * 32))(data_GT)
            data_IN = transforms.Resize(((h // 32) * 32, (w // 32) * 32))(data_IN)

        return data_IN,data_GT,img_name

    def __len__(self):
        return len(self.imgs_in)
class DatasetForInference(Dataset):
    def __init__(self, dir_path):
        self.image_paths =  glob.glob( os.path.join(dir_path, '*') )
        self.transform = transforms.Compose([
            transforms.Resize([128, 128]),
            #transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]) #transforms.ToTensor()
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        input_path = self.image_paths[index]
        input_image = Image.open(input_path).convert('RGB')
        input_image = self.transform(input_image)
        _, h, w = input_image.shape
        if (h%16 != 0) or (w%16 != 0):
            input_image = transforms.Resize(((h//16)*16, (w//16)*16))(input_image)
        return input_image #, os.path.basename(input_path)


if __name__ == '__main__':
    rootA = '/gdata2/zhuyr/Weather/Data/Haze/REVID'
    rootA_txt = '/gdata2/zhuyr/Weather/Data/RealHaze.txt'
    rootB = '/gdata2/zhuyr/Weather/Data/Snow'  #'/gdata2/zhuyr/Weather/Data/Rain/'
    rootB_txt = '/gdata2/zhuyr/Weather/Data/RealSnow.txt' #'/gdata2/zhuyr/Weather/Data/spa_jpg.txt'
    rootC = '/gdata2/zhuyr/Weather/Data/Snow'
    rootC_txt = '/gdata2/zhuyr/Weather/Data/RealSnow.txt'


    evalA_in =  '/gdata2/zhuyr/Weather/Data/Haze/REVID/Test/hazy_re/'
    evalA_gt = '/gdata2/zhuyr/Weather/Data/Haze/REVID/Test/gt_re/'

    train_set = my_dataset(rootA, rootA_txt, rootB, rootB_txt,rootC, rootC_txt,crop_size =224,
                           fix_sample_A = 10,fix_sample_B = 5,fix_sample_C = 3)
    train_loader = iter(DataLoader(train_set, batch_size=3, num_workers=4, shuffle=True, drop_last=False,pin_memory=True))
    #train_loader +
    print('len(train_loader):',len(train_loader))
    max_epoch=100
    for epoch in range(max_epoch):
        count = 0
        for train_idx, train_data in enumerate(train_loader):
            count += 1
            total_iter = epoch* len(train_loader) + count
            print('---------' * 2, iter)
            data_A, data_B, data_C = next(train_loader)
            print(data_A[0].size(),data_A[2], data_A[3])
            print(data_B[0].size(),data_B[2], data_B[3])
            print(data_C[0].size(),data_C[2], data_C[3])
    #
    #     data_A, data_B, data_C = train_data
    #     print('---------'*2,train_idx)
    #     print(data_A[0].size(),data_A[2], data_A[3])
    #     print(data_B[0].size(),data_B[2], data_B[3])
    #     print(data_C[0].size(),data_C[2], data_C[3])


    # trans_eval = transforms.Compose(
    #     [
    #         transforms.ToTensor()
    #     ])

    # evalA_set = my_dataset_eval_realH(root_in= evalA_in,root_label= evalA_gt,transform=trans_eval,fix_sample= 200 )
    # evalA_loader =  DataLoader(evalA_set, batch_size=1, num_workers=4)
    # for eval_idx, (data_in, label, name) in enumerate(evalA_loader, 0):
    #     print('---------'*2, eval_idx)
    #     print(data_in.size(), label.size())
    #
    # evalA_set = my_dataset_eval_realH(root_in= evalA_in,root_label= evalA_gt,transform=trans_eval,fix_sample= 200 )
    # evalA_loader =  DataLoader(evalA_set, batch_size=1, num_workers=4)
    # for eval_idx, (data_in, label, name) in enumerate(evalA_loader, 0):
    #     print('---------'*2, eval_idx)
    #     print(data_in.size(), label.size())

