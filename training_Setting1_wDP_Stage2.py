import os,cv2,time,torchvision,argparse,logging,sys,os,gc
import torch,math,random
import numpy as np
from tqdm import tqdm

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from torch.autograd import Variable
import torch.optim as optim

from datasets.dataset_pairs_wRandomSample import my_dataset,my_dataset_eval
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR,CosineAnnealingWarmRestarts,MultiStepLR


from utils.UTILS import compute_psnr
import loss.losses as losses
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from loss.perceptual import LossNetwork


sys.path.append(os.getcwd())
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(20)

if torch.cuda.device_count() ==8:
    MULTI_GPU = True
    print('MULTI_GPU {}:'.format(torch.cuda.device_count()), MULTI_GPU )
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1,2,3,4, 5,6,7"
    device_ids = [0, 1,2,3,4, 5,6,7]
if torch.cuda.device_count() == 4:
    MULTI_GPU = True
    print('MULTI_GPU {}:'.format(torch.cuda.device_count()), MULTI_GPU )
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1,2,3"
    device_ids = [0, 1,2,3]
if torch.cuda.device_count() == 2:
    MULTI_GPU = True
    print('MULTI_GPU {}:'.format(torch.cuda.device_count()), MULTI_GPU )
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
    device_ids = [0, 1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device ----------------------------------------:',device)

parser = argparse.ArgumentParser()
# path setting
parser.add_argument('--experiment_name', type=str,default= "training_R1K_H500_S2k_JointPre_PP_Tri") # modify the experiments name-->modify all save path
parser.add_argument('--unified_path', type=str,default=  '/gdata2/zhuyr/Weather/')
#parser.add_argument('--model_save_dir', type=str, default= )#required=True
parser.add_argument('--training_in_path', type=str,default= '/gdata2/zhuyr/Weather/Data/Snow/all_trainingData/synthetic/')
parser.add_argument('--training_gt_path', type=str,default= '/gdata2/zhuyr/Weather/Data/Snow/all_trainingData/gt/')

parser.add_argument('--training_in_pathRain', type=str,default= '/gdata2/zhuyr/Weather/Data/Rain/HeavyRain/Train/in_0917/')
parser.add_argument('--training_gt_pathRain', type=str,default= '/gdata2/zhuyr/Weather/Data/Rain/HeavyRain/Train/gt_0917/')

parser.add_argument('--training_in_pathRD', type=str,default= '/gdata2/zhuyr/Weather/Data/RainDrop/train/train/data/')#  RainDrop 1110 pairs
parser.add_argument('--training_gt_pathRD', type=str,default= '/gdata2/zhuyr/Weather/Data/RainDrop/train/train/gt/')


parser.add_argument('--writer_dir', type=str, default= '/ghome/zhuyr/UDC_codes/writer_logs/')
parser.add_argument('--logging_path', type=str, default= '/ghome/zhuyr/UDC_codes/logging/')

parser.add_argument('--eval_in_path_RD', type=str,default= '/gdata2/zhuyr/Weather/Data/RainDrop/test_a/test_a/data-re/')
parser.add_argument('--eval_gt_path_RD', type=str,default= '/gdata2/zhuyr/Weather/Data/RainDrop/test_a/test_a/gt-re/')

parser.add_argument('--eval_in_path_L', type=str,default= '/gdata2/zhuyr/Weather/Data/Snow/test/Snow100K-L/synthetic/')
parser.add_argument('--eval_gt_path_L', type=str,default= '/gdata2/zhuyr/Weather/Data/Snow/test/Snow100K-L/gt/')

parser.add_argument('--eval_in_path_Rain', type=str,default= '/gdata2/zhuyr/Weather/Data/Rain/HeavyRain/Test/in/')
parser.add_argument('--eval_gt_path_Rain', type=str,default= '/gdata2/zhuyr/Weather/Data/Rain/HeavyRain/Test/gt_re/')
#training setting
parser.add_argument('--EPOCH', type=int, default= 180)
parser.add_argument('--T_period', type=int, default= 60)  # CosineAnnealingWarmRestarts
parser.add_argument('--BATCH_SIZE', type=int, default= 10)
parser.add_argument('--Crop_patches', type=int, default= 256)
parser.add_argument('--learning_rate', type=float, default= 0.0001)
parser.add_argument('--print_frequency', type=int, default= 200)
parser.add_argument('--SAVE_Inter_Results', type=bool, default= False)
#during training
parser.add_argument('--max_psnr', type=int, default= 25)
parser.add_argument('--fix_sample', type=int, default= 10000)
parser.add_argument('--VGG_lamda', type=float, default= 0.1)

parser.add_argument('--debug', type=bool, default= False)
parser.add_argument('--lam', type=float, default= 0.1)
parser.add_argument('--flag', type=str, default= 'K1')
parser.add_argument('--pre_model', type=str,default= '/gdata2/zhuyr/Weather/training_Setting1_PP1004-2/net_epoch_88.pth')

#training setting
parser.add_argument('--base_channel', type = int, default= 20)
parser.add_argument('--num_block', type=int, default= 6)
args = parser.parse_args()


if args.debug == True:
    fix_sample = 200
else:
    fix_sample = args.fix_sample

exper_name =args.experiment_name
writer = SummaryWriter(args.writer_dir + exper_name)
if not os.path.exists(args.writer_dir):
    os.mkdir(args.writer_dir)
if not os.path.exists(args.logging_path):
    os.mkdir(args.logging_path)

unified_path = args.unified_path
SAVE_PATH =unified_path  + exper_name + '/'
if not os.path.exists(SAVE_PATH):
    os.mkdir(SAVE_PATH)
if args.SAVE_Inter_Results:
    SAVE_Inter_Results_PATH = unified_path + exper_name +'__inter_results/'
    if not os.path.exists(SAVE_Inter_Results_PATH):
        os.mkdir(SAVE_Inter_Results_PATH)

base_channel=args.base_channel
num_res = args.num_block

trans_eval = transforms.Compose(
        [
         transforms.ToTensor()
        ])
logging.info(f'begin testing! ')
print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
print("=="*50)
def check_dataset(in_path, gt_path,name ='RD'):
    print( "Check {} pairs({}) ???: {} ".format(name,len(in_path), os.listdir(in_path)==os.listdir(gt_path)) )
check_dataset(args.eval_in_path_RD,args.eval_gt_path_RD,'val-RD' )
check_dataset(args.eval_in_path_Rain,args.eval_gt_path_Rain,'val-Rain' )
check_dataset(args.eval_in_path_L,args.eval_gt_path_L,'val-Snow-L' )
check_dataset(args.training_in_path,args.training_gt_path,'Train_Snow' )
check_dataset(args.training_in_pathRain,args.training_gt_pathRain,'Train_Rain' )
check_dataset(args.training_in_pathRD,args.training_gt_pathRD,'Train_RD' )
print("=="*50)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def test(net,eval_loader, save_model ,epoch =1,max_psnr_val=26 ,Dname = 'S',flag = [1,0,0]):
    net.to('cuda:0')
    net.eval()
    net.load_state_dict(torch.load(save_model), strict=True)

    st = time.time()
    with torch.no_grad():
        eval_output_psnr = 0.0
        eval_input_psnr = 0.0
        for index, (data_in, label, name) in enumerate(eval_loader, 0):#enumerate(tqdm(eval_loader), 0):
            inputs = Variable(data_in).to('cuda:0')
            labels = Variable(label).to('cuda:0')

            outputs = net(inputs,flag=flag)
            eval_input_psnr += compute_psnr(inputs, labels)
            eval_output_psnr += compute_psnr(outputs, labels)
        Final_output_PSNR = eval_output_psnr / len(eval_loader)
        Final_input_PSNR = eval_input_psnr / len(eval_loader)
        writer.add_scalars(exper_name + '/testing', {'eval_PSNR_Output': eval_output_psnr / len(eval_loader),
                                                     'eval_PSNR_Input': eval_input_psnr / len(eval_loader), }, epoch)
        if Final_output_PSNR > max_psnr_val:  #just save better model
            max_psnr_val = Final_output_PSNR
        print("epoch:{}---------Dname:{}--------------[Num_eval:{} In_PSNR:{}  Out_PSNR:{}]--------max_psnr_val:{}:-----cost time;{}".format(epoch, Dname,len(eval_loader),round(Final_input_PSNR, 2),
                                                                                        round(Final_output_PSNR, 2), round(max_psnr_val, 2),time.time() -st))
    return max_psnr_val


def save_imgs_for_visual(path,inputs,labels,outputs):
    torchvision.utils.save_image([inputs.cpu()[0], labels.cpu()[0], outputs.cpu()[0]], path,nrow=3, padding=0)

def get_training_data(fix_sample=fix_sample, Crop_patches=args.Crop_patches):
    rootA_in = args.training_in_path
    rootA_label = args.training_gt_path
    rootB_in = args.training_in_pathRain
    rootB_label = args.training_gt_pathRain
    rootC_in = args.training_in_pathRD
    rootC_label = args.training_gt_pathRD
    train_datasets = my_dataset(rootA_in, rootA_label, rootB_in, rootB_label,rootC_in, rootC_label,crop_size =Crop_patches,
                                fix_sample_A = fix_sample, fix_sample_B = fix_sample,fix_sample_C = fix_sample)
    train_loader = DataLoader(dataset=train_datasets, batch_size=args.BATCH_SIZE, num_workers= 6 ,shuffle=True)
    print('len(train_loader):' ,len(train_loader))
    return train_loader

def get_eval_data(val_in_path=args.eval_in_path_L,val_gt_path =args.eval_gt_path_L ,trans_eval=trans_eval):
    eval_data = my_dataset_eval(
        root_in=val_in_path, root_label =val_gt_path, transform=trans_eval,fix_sample= 500 )
    eval_loader = DataLoader(dataset=eval_data, batch_size=1, num_workers=4)
    return eval_loader
def print_param_number(net):
    print('#generator parameters:', sum(param.numel() for param in net.parameters()))
    Total_params = 0
    Trainable_params = 0

    for param in net.parameters():
        mulValue = np.prod(param.size())
        Total_params += mulValue
        if param.requires_grad:
            Trainable_params += mulValue
    print(f'Total params: {Total_params}')
    print(f'Trainable params: {Trainable_params}')


if __name__ == '__main__':
    if args.flag == 'K1':
        from networks.Network_Stage2_K1_Flag import UNet
    elif args.flag == 'K3':
        from networks.Network_Stage2_K3_Flag import UNet

    net = UNet(base_channel=base_channel, num_res=num_res)
    net_eval = UNet(base_channel=base_channel, num_res=num_res)
    pretrained_model = torch.load(args.pre_model)
    net.load_state_dict(pretrained_model, strict=False)

    net = nn.DataParallel(net, device_ids= device_ids)
    net.to(device)
    print_param_number(net)

    train_loader = get_training_data()
    eval_loader_RD = get_eval_data(val_in_path=args.eval_in_path_RD,val_gt_path =args.eval_gt_path_RD)
    eval_loader_Rain = get_eval_data(val_in_path=args.eval_in_path_Rain, val_gt_path=args.eval_gt_path_Rain)
    eval_loader_L = get_eval_data(val_in_path=args.eval_in_path_L, val_gt_path =args.eval_gt_path_L)


    for name, param in net.named_parameters():
        if "B1" not in name and "B2" not in name and "B3" not in name:
            param.requires_grad = False

    optimizerG_B1 = optim.Adam(net.parameters(), lr=args.learning_rate,betas=(0.9,0.999))
    scheduler_B1 = CosineAnnealingWarmRestarts(optimizerG_B1, T_0=args.T_period,  T_mult=1) #MultiStepLR(optimizerG_B1, milestones=[5,20,40,60,80], gamma=0.5)# args.milestep   [5,20,40,60,80]
    #
    optimizerG_B1 = nn.DataParallel(optimizerG_B1, device_ids=device_ids)
    scheduler_B1 = nn.DataParallel(scheduler_B1, device_ids=device_ids)

    loss_char= losses.CharbonnierLoss()

    vgg = models.vgg16(pretrained=False)
    vgg.load_state_dict(torch.load('/gdata2/zhuyr/VGG/vgg16-397923af.pth'))
    vgg_model = vgg.features[:16]
    vgg_model = vgg_model.to(device)
    for param in vgg_model.parameters():
        param.requires_grad = False
    loss_network = LossNetwork(vgg_model)
    loss_network.eval()


    step =0
    max_psnr_val_L = args.max_psnr
    max_psnr_val_Rain = args.max_psnr
    max_psnr_val_RD = args.max_psnr

    total_lossA = 0.0
    total_lossB = 0.0
    total_lossC  = 0.0

    total_loss1 = 0.0
    total_loss2 = 0.0
    total_loss3 = 0.0
    total_loss4 = 0.0
    total_loss5 = 0.0
    total_loss6 = 0.0
    total_loss = 0.0

    input_PSNR_all_A = 0
    train_PSNR_all_A = 0
    input_PSNR_all_B = 0
    train_PSNR_all_B = 0
    input_PSNR_all_C = 0
    train_PSNR_all_C = 0
    Frequncy_eval_save = len(train_loader)

    iter_nums = 0
    for epoch in range(args.EPOCH):
        scheduler_B1.module.step(epoch)

        st = time.time()
        for i,train_data in enumerate(train_loader,0):#   (data_in, label)  ----- train_data
            #data_A, data_B = train_data
            data_A, data_B, data_C = train_data
            if i ==0:
                print("data_A.size(),in_GT:",data_A[0].size(), data_A[1].size())  # Snow
                print("data_B.size(),in_GT:", data_B[0].size(), data_B[1].size()) # Rain
                print("data_C.size(),in_GT:", data_C[0].size(), data_C[1].size()) # RD

            iter_nums +=1
            net.train()


            inputs_A = Variable(data_A[0]).to(device)
            labels_A = Variable(data_A[1]).to(device)
            inputs_B = Variable(data_B[0]).to(device)
            labels_B = Variable(data_B[1]).to(device)
            inputs_C = Variable(data_C[0]).to(device)
            labels_C = Variable(data_C[1]).to(device)
            #--------------------------------------------optimizerG_B1---------------------------------------------#


            # ============================== data A  ============================== #
            net.module.zero_grad()
            optimizerG_B1.module.zero_grad()

            train_output_A = net(inputs_A, flag = [1,0,0])
            input_PSNR_A = compute_psnr(inputs_A, labels_A)
            trian_PSNR_A = compute_psnr(train_output_A, labels_A)

            loss1 = F.smooth_l1_loss(train_output_A, labels_A) +  args.VGG_lamda * loss_network(train_output_A, labels_A)
            loss2 = args.lam * sum([abs(i) for i in net.module.getIndicators_B1()]) /1000
            g_lossA = loss1 + loss2
            total_lossA += g_lossA.item()
            input_PSNR_all_A += input_PSNR_A
            train_PSNR_all_A += trian_PSNR_A

            g_lossA.backward()
            optimizerG_B1.module.step()

            # ============================== data B  ============================== #
            net.module.zero_grad()
            optimizerG_B1.module.zero_grad()

            train_output_B = net(inputs_B, flag = [0,1,0])
            input_PSNR_B = compute_psnr(inputs_B, labels_B)
            trian_PSNR_B = compute_psnr(train_output_B, labels_B)

            loss3 = F.smooth_l1_loss(train_output_B, labels_B) + args.VGG_lamda * loss_network(train_output_B, labels_B)
            loss4 = args.lam * sum([abs(i) for i in net.module.getIndicators_B2()]) / 1000

            g_lossB = loss3 + loss4
            total_lossB += g_lossB.item()

            input_PSNR_all_B += input_PSNR_B
            train_PSNR_all_B += trian_PSNR_B

            g_lossB.backward()
            optimizerG_B1.module.step()
            # ============================== data C  ============================== #
            net.module.zero_grad()
            optimizerG_B1.module.zero_grad()

            train_output_C = net(inputs_C,flag = [0, 0, 1])
            input_PSNR_C = compute_psnr(inputs_C, labels_C)
            trian_PSNR_C = compute_psnr(train_output_C, labels_C)

            loss5 = F.smooth_l1_loss(train_output_C, labels_C) +  args.VGG_lamda * loss_network(train_output_C, labels_C)
            loss6 = args.lam * sum([abs(i) for i in net.module.getIndicators_B3()]) /1000

            g_lossC =  loss5 + loss6
            total_lossC += g_lossC.item()
            input_PSNR_all_C += input_PSNR_C
            train_PSNR_all_C += trian_PSNR_C

            g_lossC.backward()
            optimizerG_B1.module.step()

            g_loss = g_lossA + g_lossB + g_lossC

            #-----------------------------------------------------------------------------------------#
            total_loss += g_loss.item()
            total_loss1 += loss1.item()
            total_loss2 += loss2.item()
            total_loss3 += loss3.item()
            total_loss4 += loss4.item()
            total_loss5 += loss5.item()
            total_loss6 += loss6.item()

            Percent_B1 = torch.mean((torch.tensor(net.module.getIndicators_B1()) >= .1).float())
            Percent_B2 = torch.mean((torch.tensor(net.module.getIndicators_B2()) >= .1).float())
            Percent_B3 = torch.mean((torch.tensor(net.module.getIndicators_B3()) >= .1).float())
            
            Percent_B1_1 = torch.mean((torch.tensor(net.module.getIndicators_B1()) >= .2).float())
            Percent_B2_1 = torch.mean((torch.tensor(net.module.getIndicators_B2()) >= .2).float())
            Percent_B3_1 = torch.mean((torch.tensor(net.module.getIndicators_B3()) >= .2).float())
            
            Percent_B1_2 = torch.mean((torch.tensor(net.module.getIndicators_B1()) >= .4).float())
            Percent_B2_2 = torch.mean((torch.tensor(net.module.getIndicators_B2()) >= .4).float())
            Percent_B3_2 = torch.mean((torch.tensor(net.module.getIndicators_B3()) >= .4).float())


            if (i+1) % args.print_frequency ==0 and i >1:
                writer.add_scalars(exper_name +'/training_PSNR' ,{'PSNR_Output_A': train_PSNR_all_A / iter_nums,'PSNR_Input_A': input_PSNR_all_A / iter_nums,
                                                             'PSNR_Output_B': train_PSNR_all_B / iter_nums,'PSNR_Input_B': input_PSNR_all_B / iter_nums
                                                             ,'PSNR_Output_C': train_PSNR_all_C / iter_nums,'PSNR_Input_C': input_PSNR_all_C / iter_nums} , iter_nums)
                writer.add_scalars(exper_name +'/training_Loss' ,{'total_loss_A': total_lossA / iter_nums,'total_loss_B': total_lossB / iter_nums,'total_loss_C': total_lossC/ iter_nums,
                                                             'loss1': total_loss1 / iter_nums, 'loss2': total_loss2 / iter_nums,'loss3': total_loss3 / iter_nums,'loss4': total_loss4 / iter_nums,
                                                             'loss5': total_loss5 / iter_nums, 'loss6': total_loss6 / iter_nums, 'total loss': total_loss / iter_nums} , iter_nums)
                writer.add_scalar(exper_name + '/Percent Weights Activated_B1',
                                  torch.mean((torch.tensor(net.module.getIndicators_B1()) >= .1).float()), iter_nums)
                writer.add_scalar(exper_name + '/Percent Weights Activated_B2',
                                  torch.mean((torch.tensor(net.module.getIndicators_B2()) >= .1).float()), iter_nums)
                writer.add_scalar(exper_name + '/Percent Weights Activated_B3',
                                  torch.mean((torch.tensor(net.module.getIndicators_B3()) >= .1).float()), iter_nums)
                print(
                    "[epoch:%d / EPOCH :%d],[%d / %d], [lr: %.7f ],[loss1:%.5f,loss2:%.5f,loss3:%.5f,loss4:%.5f,loss5:%.5f,loss6:%.5f, avg_lossA:%.5f, avg_lossB:%.5f, avg_lossC:%.5f, avg_loss:%.5f],"
                    "[in_PSNR_A: %.3f, out_PSNR_A: %.3f],[in_PSNR_B: %.3f, out_PSNR_B: %.3f],[in_PSNR_C: %.3f, out_PSNR_C: %.3f],"
                    "[PercentB1: %.3f, PercentB2: %.3f, PercentB3: %.3f], time: %.3f" %
                    (epoch,args.EPOCH, i + 1, len(train_loader), optimizerG_B1.module.param_groups[0]["lr"],  loss1.item(),
                     loss2.item(),loss3.item(),loss4.item(), loss5.item(),loss6.item(),total_lossA / iter_nums,total_lossB / iter_nums, total_lossC / iter_nums,total_loss / iter_nums,
                     input_PSNR_A, trian_PSNR_A, input_PSNR_B, trian_PSNR_B, input_PSNR_C, trian_PSNR_C, Percent_B1.item(), Percent_B2.item(), Percent_B3.item(),time.time() - st))
                print("[Threshold [0.1],  PercentB1: %.3f, PercentB2: %.3f, PercentB3: %.3f]" % (Percent_B1.item(), Percent_B2.item(), Percent_B3.item()))
                print("[Threshold [0.2],  PercentB1: %.3f, PercentB2: %.3f, PercentB3: %.3f]"%(Percent_B1_1.item(), Percent_B2_1.item(), Percent_B3_1.item()))
                print("[Threshold [0.4],  PercentB1: %.3f, PercentB2: %.3f, PercentB3: %.3f]"%(Percent_B1_2.item(), Percent_B2_2.item(), Percent_B3_2.item()))

                st = time.time()
            # if args.SAVE_Inter_Results:
            #     save_path = SAVE_Inter_Results_PATH + str(iter_nums) + '.jpg'
            #     save_imgs_for_visual(save_path, inputs, labels, train_output)
        save_model = SAVE_PATH  + 'net_epoch_{}.pth'.format(epoch)
        torch.save(net.module.state_dict(),save_model)

        max_psnr_val_L = test(net= net_eval, save_model = save_model,  eval_loader = eval_loader_L,epoch=epoch,max_psnr_val = max_psnr_val_L, Dname = 'Snow-L',flag = [1,0,0])
        max_psnr_val_Rain = test(net=net_eval, save_model = save_model, eval_loader = eval_loader_Rain, epoch=epoch, max_psnr_val=max_psnr_val_Rain, Dname= 'HRain',flag = [0,1,0])
        max_psnr_val_RD = test(net=net_eval, save_model  = save_model, eval_loader = eval_loader_RD, epoch=epoch, max_psnr_val=max_psnr_val_RD, Dname= 'RD',flag = [0,0,1] )
