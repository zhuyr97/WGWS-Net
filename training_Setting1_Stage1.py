import time,torchvision,argparse,sys,os
import torch,random
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from torch.autograd import Variable
import torch.optim as optim

from datasets.dataset_pairs_wRandomSample_Triplet import my_dataset,my_dataset_eval
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR,CosineAnnealingWarmRestarts
from networks.Network_Stage1 import UNet
from utils.UTILS import compute_psnr
import loss.losses as losses
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from loss.perceptual import LossNetwork

sys.path.append(os.getcwd())
# 设置随机数种子
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(20)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device ----------------------------------------:',device)

parser = argparse.ArgumentParser()
# path setting
parser.add_argument('--experiment_name', type=str,default= "training_R1400_H500_S100K_PP2") # modify the experiments name-->modify all save path
parser.add_argument('--unified_path', type=str,default=  '/gdata2/zhuyr/Weather/')
#parser.add_argument('--model_save_dir', type=str, default= )#required=True
parser.add_argument('--training_in_path', type=str,default= '/gdata2/zhuyr/Weather/Data/Snow/all_trainingData/synthetic/')
parser.add_argument('--training_gt_path', type=str,default= '/gdata2/zhuyr/Weather/Data/Snow/all_trainingData/gt/')

parser.add_argument('--training_in_pathRain', type=str,default= '/gdata2/zhuyr/Weather/Data/Rain/HeavyRain/Train/in_0917/')
parser.add_argument('--training_gt_pathRain', type=str,default= '/gdata2/zhuyr/Weather/Data/Rain/HeavyRain/Train/gt_0917/')

parser.add_argument('--training_in_pathRD', type=str,default= '/gdata2/zhuyr/Weather/Data/RainDrop/train/train/data/')
parser.add_argument('--training_gt_pathRD', type=str,default= '/gdata2/zhuyr/Weather/Data/RainDrop/train/train/gt/')


parser.add_argument('--writer_dir', type=str, default= '/ghome/zhuyr/UDC_codes/writer_logs/')

parser.add_argument('--eval_in_path_RD', type=str,default= '/gdata2/zhuyr/Weather/Data/RainDrop/test_a/test_a/data-re/')
parser.add_argument('--eval_gt_path_RD', type=str,default= '/gdata2/zhuyr/Weather/Data/RainDrop/test_a/test_a/gt-re/')

parser.add_argument('--eval_in_path_L', type=str,default= '/gdata2/zhuyr/Weather/Data/Snow/test/Snow100K-L/synthetic/')
parser.add_argument('--eval_gt_path_L', type=str,default= '/gdata2/zhuyr/Weather/Data/Snow/test/Snow100K-L/gt/')

parser.add_argument('--eval_in_path_Rain', type=str,default= '/gdata2/zhuyr/Weather/Data/Rain/HeavyRain/Test/in/')
parser.add_argument('--eval_gt_path_Rain', type=str,default= '/gdata2/zhuyr/Weather/Data/Rain/HeavyRain/Test/gt_re/')


#training setting
parser.add_argument('--EPOCH', type=int, default= 100)
parser.add_argument('--T_period', type=int, default= 50)  # CosineAnnealingWarmRestarts
parser.add_argument('--BATCH_SIZE', type=int, default= 4)
parser.add_argument('--Crop_patches', type=int, default= 200)
parser.add_argument('--learning_rate', type=float, default= 0.0002)
parser.add_argument('--print_frequency', type=int, default= 50)
parser.add_argument('--SAVE_Inter_Results', type=bool, default= False)
#during training
parser.add_argument('--max_psnr', type=int, default= 10)
parser.add_argument('--fix_sample', type=int, default= 9000)
parser.add_argument('--lam_VGG', type=float, default= 0.1)

parser.add_argument('--debug', type=bool, default= False)
parser.add_argument('--addition_loss', type=str, default= 'VGG')
parser.add_argument('--depth_loss', type=bool, default= False) 
parser.add_argument('--lam_DepthLoss', type=float, default= 0.1)

parser.add_argument('--Aug_regular', type=bool, default= False)
#training setting
parser.add_argument('--base_channel', type = int, default= 24)
parser.add_argument('--num_block', type=int, default= 6)
args = parser.parse_args()


if args.debug ==True:
    fix_sampleA = 400
    fix_sampleB = 400
    fix_sampleC = 400
    print_frequency = 10

else:
    fix_sampleA = args.fix_sample
    fix_sampleB = args.fix_sample
    fix_sampleC = args.fix_sample
    print_frequency =args.print_frequency

exper_name =args.experiment_name
writer = SummaryWriter(args.writer_dir + exper_name)
if not os.path.exists(args.writer_dir):
    os.mkdir(args.writer_dir)

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


def test(net,eval_loader,epoch =1,max_psnr_val=26 ,Dname = 'S'):
    net.eval()
    with torch.no_grad():
        eval_output_psnr = 0.0
        eval_input_psnr = 0.0
        st = time.time()
        for index, (data_in, label, name) in enumerate(eval_loader, 0):#enumerate(tqdm(eval_loader), 0):
            inputs = Variable(data_in).to(device)
            labels = Variable(label).to(device)
            outputs = net(inputs)
            eval_input_psnr += compute_psnr(inputs, labels)
            eval_output_psnr += compute_psnr(outputs, labels)
        Final_output_PSNR = eval_output_psnr / len(eval_loader)
        Final_input_PSNR = eval_input_psnr / len(eval_loader)
        writer.add_scalars(exper_name + '/testing', {'eval_PSNR_Output': eval_output_psnr / len(eval_loader),
                                                     'eval_PSNR_Input': eval_input_psnr / len(eval_loader), }, epoch)
        if Final_output_PSNR > max_psnr_val:
            max_psnr_val = Final_output_PSNR
        print("epoch:{}---------Dname:{}--------------[Num_eval:{} In_PSNR:{}  Out_PSNR:{}]--------max_psnr_val:{}, cost time: {}".format(epoch, Dname,len(eval_loader),round(Final_input_PSNR, 2),
                                                                                        round(Final_output_PSNR, 2), round(max_psnr_val, 2), time.time() -st ))
    return max_psnr_val

def save_imgs_for_visual(path,inputs,labels,outputs):
    torchvision.utils.save_image([inputs.cpu()[0], labels.cpu()[0], outputs.cpu()[0]], path,nrow=3, padding=0)

def get_training_data(fix_sampleA= fix_sampleA, fix_sampleB= fix_sampleB,fix_sampleC= fix_sampleC, Crop_patches=args.Crop_patches):
    rootA_in = args.training_in_path
    rootA_label = args.training_gt_path
    rootB_in = args.training_in_pathRain
    rootB_label = args.training_gt_pathRain
    rootC_in = args.training_in_pathRD
    rootC_label = args.training_gt_pathRD
    train_datasets = my_dataset(rootA_in, rootA_label, rootB_in, rootB_label,rootC_in, rootC_label,crop_size =Crop_patches,
                                fix_sample_A = fix_sampleA, fix_sample_B = fix_sampleB,fix_sample_C = fix_sampleC,regular_aug=args.Aug_regular)
    train_loader = DataLoader(dataset=train_datasets, batch_size=args.BATCH_SIZE, num_workers= 8 ,shuffle=True)
    print('len(train_loader):' ,len(train_loader))
    return train_loader

def get_eval_data(val_in_path=args.eval_in_path_Rain,val_gt_path =args.eval_gt_path_Rain ,trans_eval=trans_eval):
    eval_data = my_dataset_eval(
        root_in=val_in_path, root_label =val_gt_path, transform=trans_eval,fix_sample= 500 )
    eval_loader = DataLoader(dataset=eval_data, batch_size=1, num_workers= 4)
    return eval_loader
def print_param_number(net):
    print('#generator parameters:', sum(param.numel() for param in net.parameters()))

if __name__ == '__main__':
    net =UNet(base_channel=base_channel, num_res=num_res)
    net.to(device)
    print_param_number(net)

    train_loader = get_training_data()
    eval_loader_RD = get_eval_data(val_in_path=args.eval_in_path_RD,val_gt_path =args.eval_gt_path_RD)
    eval_loader_Rain = get_eval_data(val_in_path=args.eval_in_path_Rain, val_gt_path=args.eval_gt_path_Rain)
    eval_loader_L = get_eval_data(val_in_path=args.eval_in_path_L, val_gt_path =args.eval_gt_path_L)

    optimizerG = optim.Adam(net.parameters(), lr=args.learning_rate,betas=(0.9,0.999))
    scheduler = CosineAnnealingWarmRestarts(optimizerG, T_0=args.T_period,  T_mult=1) #ExponentialLR(optimizerG, gamma=0.98)
    # Losses
    loss_char= losses.CharbonnierLoss()
    criterion_depth = losses.depth_loss()
    # 1
    #vgg = models.vgg16(pretrained=False)
    # 2
    vgg = models.vgg16(pretrained=False)
    vgg.load_state_dict(torch.load('/gdata2/zhuyr/VGG/vgg16-397923af.pth'))
    vgg_model = vgg.features[:16]
    vgg_model = vgg_model.to(device)
    for param in vgg_model.parameters():
        param.requires_grad = False
    loss_network = LossNetwork(vgg_model)
    loss_network.eval()
    
    
    step =0
    max_psnr_val_RD= args.max_psnr
    max_psnr_val_Rain = args.max_psnr
    max_psnr_val_S = args.max_psnr
    max_psnr_val_M = args.max_psnr
    max_psnr_val_L = args.max_psnr

    total_loss = 0.0
    total_loss1 = 0.0
    total_loss2 = 0.0
    total_loss3 = 0.0
    input_PSNR_all = 0
    train_PSNR_all = 0
    Frequncy_eval_save = len(train_loader)

    iter_nums = 0
    for epoch in range(args.EPOCH):
        scheduler.step(epoch)
        st = time.time()
        for i,train_data in enumerate(train_loader,0):
            data_A, data_B, data_C = train_data
            data_in = torch.cat([data_A[0],data_B[0],data_C[0]],dim=0)
            label = torch.cat([data_A[1],data_B[1],data_C[1]],dim=0)
            if i == 0:
                print("Check data: data.size: {} ,in_GT_mask: {}".format(data_in.size(),label.size()))
            iter_nums +=1
            net.train()
            net.zero_grad()
            optimizerG.zero_grad()
            inputs = Variable(data_in).to(device)
            labels = Variable(label).to(device)

            train_output= net(inputs)
            input_PSNR = compute_psnr(inputs, labels)
            trian_PSNR = compute_psnr(train_output, labels)

            loss1 = F.smooth_l1_loss(train_output, labels)
            if args.addition_loss == 'VGG':
                loss2 =  args.lam_VGG * loss_network(train_output, labels)
            else:
                loss2 = 0.01 *loss1

            if args.depth_loss :
                loss3 = args.lam_DepthLoss * criterion_depth(train_output, labels)
                g_loss = loss1 + loss2 + loss3
            else:
                loss3 = loss1
                g_loss = loss1 + loss2

            total_loss += g_loss.item()
            total_loss1 += loss1.item()
            total_loss2 += loss2.item()
            total_loss3 += loss3.item()

            input_PSNR_all += input_PSNR
            train_PSNR_all += trian_PSNR
            g_loss.backward()
            optimizerG.step()
            if (i+1) % print_frequency ==0 and i >1:
                writer.add_scalars(exper_name +'/training' ,{'PSNR_Output': train_PSNR_all / iter_nums,'PSNR_Input': input_PSNR_all / iter_nums, } , iter_nums)
                writer.add_scalars(exper_name +'/training' ,{'total_loss': total_loss / iter_nums,'loss1_char': total_loss1 / iter_nums, 'loss2': total_loss2 / iter_nums,'loss3': total_loss3 / iter_nums,} , iter_nums)
                print(
                    "epoch:%d,[%d / %d], [lr: %.7f ],[loss:%.5f,loss1:%.5f,loss2:%.5f,loss3:%.5f, avg_loss:%.5f],[in_PSNR: %.3f, out_PSNR: %.3f],time:%.3f" %
                    (epoch, i + 1, len(train_loader), optimizerG.param_groups[0]["lr"], g_loss.item(), loss1.item(),
                     loss2.item(), loss3.item(), total_loss / iter_nums, input_PSNR, trian_PSNR, time.time() - st))
                st = time.time()

            if args.SAVE_Inter_Results:
                save_path = SAVE_Inter_Results_PATH + str(iter_nums) + '.jpg'
                save_imgs_for_visual(save_path, inputs, labels, train_output)

        torch.save(net.state_dict(),
                   SAVE_PATH  + 'net_epoch_{}.pth'.format(epoch))

        max_psnr_val_RD = test(net= net,eval_loader = eval_loader_RD,epoch=epoch,max_psnr_val = max_psnr_val_RD, Dname = 'RD')
        max_psnr_val_Rain = test(net=net, eval_loader = eval_loader_Rain, epoch=epoch, max_psnr_val=max_psnr_val_Rain, Dname='HRain')
        max_psnr_val_L = test(net=net, eval_loader=eval_loader_L, epoch=epoch, max_psnr_val=max_psnr_val_L, Dname='L')