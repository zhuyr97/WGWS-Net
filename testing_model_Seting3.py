import os,cv2,time,torchvision,argparse,logging,sys,os,gc
import torch,math,random
import numpy as np
from tqdm import tqdm

from torch.utils.data import Dataset,DataLoader
from torch.autograd import Variable
import matplotlib.image as img
from datasets.dataset_pairs_wRandomSample_Triplet_txt import  my_dataset_eval, my_dataset_eval_realH
import torchvision.transforms as transforms
from utils.UTILS import compute_psnr,compute_ssim

sys.path.append(os.getcwd())
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(20)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
parser = argparse.ArgumentParser()


parser.add_argument('--eval_in_path_Haze', type=str,default= '/gdata2/zhuyr/Weather/Data/Haze/REVID/Test/hazy_re/')
parser.add_argument('--eval_gt_path_Haze', type=str,default= '/gdata2/zhuyr/Weather/Data/Haze/REVID/Test/gt_re/')

parser.add_argument('--eval_in_path_Rain', type=str,default= '/gdata2/zhuyr/Weather/Data/Rain/RealRain/Testing/real_test_1000/rain_jpg/')
parser.add_argument('--eval_gt_path_Rain', type=str,default= '/gdata2/zhuyr/Weather/Data/Rain/RealRain/Testing/real_test_1000/gt_jpg/')

parser.add_argument('--eval_in_path_L', type=str,default= '/gdata2/zhuyr/Weather/Data/Snow/RealSnow/video2imgs_IN_testing_re/')
parser.add_argument('--eval_gt_path_L', type=str,default= '/gdata2/zhuyr/Weather/Data/Snow/RealSnow/video2imgs_GT_testing_re/')

parser.add_argument('--eval_in_path_M', type=str,default= '/gdata2/zhuyr/Weather/Data/Snow/test/Snow100K-M/synthetic/')
parser.add_argument('--eval_gt_path_M', type=str,default= '/gdata2/zhuyr/Weather/Data/Snow/test/Snow100K-M/gt/')

parser.add_argument('--eval_in_path_S', type=str,default= '/gdata2/zhuyr/Weather/Data/Snow/test/Snow100K-S/synthetic/')
parser.add_argument('--eval_gt_path_S', type=str,default= '/gdata2/zhuyr/Weather/Data/Snow/test/Snow100K-S/gt/')


parser.add_argument('--eval_in_path_realSnow', type=str,default= '/gdata2/zhuyr/Weather/Data/Snow/test_realistic/')
parser.add_argument('--eval_gt_path_realSnow', type=str,default= '/gdata2/zhuyr/Weather/Data/Snow/test_realistic/')

parser.add_argument('--eval_in_path_realRain', type=str,default= '/gdata2/zhuyr/Weather/Data/RealRain300/')
parser.add_argument('--eval_gt_path_realRain', type=str,default= '/gdata2/zhuyr/Weather/Data/RealRain300/')

parser.add_argument('--eval_in_path_realHaze', type=str,default= '/gdata2/zhuyr/Weather/Data/Haze/real_world_hazy_imgs/')
parser.add_argument('--eval_gt_path_realHaze', type=str,default= '/gdata2/zhuyr/Weather/Data/Haze/real_world_hazy_imgs/')


parser.add_argument('--model_path', type=str,default= '/ghome/zhuyr/WGWSNet/ckpt/')
parser.add_argument('--model_name', type=str,default= 'Setting3_K1.pth')
parser.add_argument('--save_path', type=str,default= '/ghome/zhuyr/WGWSNet/results/')

parser.add_argument('--flag', type=str, default= 'K1')
parser.add_argument('--base_channel', type = int, default= 20)
parser.add_argument('--num_block', type=int, default= 6)
args = parser.parse_args()

trans_eval = transforms.Compose(
        [
         transforms.ToTensor()
        ])

if not os.path.exists(args.save_path):
    os.mkdir(args.save_path)
    
def get_eval_data(val_in_path=args.eval_in_path_S,val_gt_path =args.eval_gt_path_S ,trans_eval=trans_eval):
    eval_data = my_dataset_eval(
        root_in=val_in_path, root_label =val_gt_path, transform=trans_eval,fix_sample= 20000 )
    eval_loader = DataLoader(dataset=eval_data, batch_size=1, num_workers=4)
    return eval_loader
def get_eval_H_data(val_in_path=args.eval_in_path_Rain,val_gt_path =args.eval_gt_path_Rain ,trans_eval=trans_eval):
    eval_data = my_dataset_eval_realH(
        root_in=val_in_path, root_label =val_gt_path, transform=trans_eval,fix_sample=  20000)
    eval_loader = DataLoader(dataset=eval_data, batch_size=1, num_workers= 4)
    return eval_loader

def test(net,eval_loader,Dname = 'S',flag = [1,0,0],model_flag= args.flag,save_results_path=args.save_path):
    net.to(device)
    net.eval()
    st = time.time()
    with torch.no_grad():
        eval_output_psnr = 0.0
        eval_input_psnr = 0.0
        eval_output_ssim = 0.0
        eval_input_ssim = 0.0
        final_save_path = save_results_path + 'Setting3-'+ model_flag + '-'+ Dname +'/'
        if not os.path.exists(final_save_path):
            os.mkdir(final_save_path)
        for index, (data_in, label, name) in enumerate(tqdm(eval_loader), 0):# enumerate(eval_loader, 0):
            inputs = Variable(data_in).to(device)
            labels = Variable(label).to(device)

            if model_flag == 'S1':
                outputs = net(inputs)
            else:
                outputs = net(inputs, flag=flag)
            eval_input_psnr += compute_psnr(inputs, labels)
            eval_output_psnr += compute_psnr(outputs, labels)
            
            eval_output_ssim += compute_ssim(outputs, labels)
            eval_input_ssim += compute_ssim(inputs, labels)
            
            out_eval_np = np.squeeze(torch.clamp(outputs, 0., 1.).cpu().detach().numpy()).transpose((1,2,0))

            img.imsave(final_save_path+name[0], np.uint8(out_eval_np * 255.))

        Final_output_PSNR = eval_output_psnr / len(eval_loader)
        Final_input_PSNR = eval_input_psnr / len(eval_loader)
        Final_output_SSIM = eval_output_ssim / len(eval_loader)
        Final_input_SSIM = eval_input_ssim / len(eval_loader)

        print("Dname:{}--------------[Num_eval:{} In_PSNR:{}  Out_PSNR:{},In_SSIM:{}  Out_SSIM:{}]:-----cost time;{}".format(
                Dname,len(eval_loader),round(Final_input_PSNR, 5),round(Final_output_PSNR, 5),round(Final_input_SSIM, 5),
                round(Final_output_SSIM, 5),time.time() - st))
def print_indictor(indictor):
    indictor_list = []
    for i in range(len(indictor)):
        indictor_list.append(indictor[i].item())
    indictor_array = np.array(indictor_list)
    print('indictor_array---ori:',indictor_array)

    x = np.zeros_like(indictor_array)
    y = np.ones_like(indictor_array)
    out = np.where(indictor_array>0.1, y,x)
    print('indictor_array---Binary out:',list(out))
if __name__ == '__main__':
    if args.flag == 'K1':
        from networks.Network_Stage2_K1_Flag import UNet
    elif args.flag == 'K3':
        from networks.Network_Stage2_K3_Flag import UNet
    elif args.flag == 'S1':
        from networks.Network_Stage1 import UNet

    net = UNet(base_channel=args.base_channel, num_res=args.num_block)
    

    pre_path = args.model_path
    index = 0

    model_name = args.model_name
    pretrained_model = torch.load(pre_path + model_name)
    net.load_state_dict(pretrained_model, strict=True)
    
    
    if args.flag != 'S1':
        indictor1 = net.getIndicators_B1()
        indictor2 = net.getIndicators_B2()
        indictor3 = net.getIndicators_B3()

        Percent_B1 = torch.mean((torch.tensor(net.getIndicators_B1()) >= .05).float())
        Percent_B2 = torch.mean((torch.tensor(net.getIndicators_B2()) >= .05).float())
        Percent_B3 = torch.mean((torch.tensor(net.getIndicators_B3()) >= .05).float())
        Percent_B1_1 = torch.mean((torch.tensor(net.getIndicators_B1()) >= .1).float())
        Percent_B2_1 = torch.mean((torch.tensor(net.getIndicators_B2()) >= .1).float())
        Percent_B3_1 = torch.mean((torch.tensor(net.getIndicators_B3()) >= .1).float())
        Percent_B1_2 = torch.mean((torch.tensor(net.getIndicators_B1()) >= .2).float())
        Percent_B2_2 = torch.mean((torch.tensor(net.getIndicators_B2()) >= .2).float())
        Percent_B3_2 = torch.mean((torch.tensor(net.getIndicators_B3()) >= .2).float())
        print("Snow (Expansion Ratios) || Percent_B1 0.05: {} |  0.1: {} | 0.15: {} ".format(Percent_B1, Percent_B1_1, Percent_B1_2))
        print("Rain (Expansion Ratios) || Percent_B2 0.05: {} |  0.1: {} | 0.15: {} ".format(Percent_B2, Percent_B2_1, Percent_B2_2))
        print("Haze (Expansion Ratios) || Percent_B3 0.05: {} |  0.1: {} | 0.15: {} ".format(Percent_B3, Percent_B3_1, Percent_B3_2))
        
    eval_loader_Haze = get_eval_H_data(val_in_path=args.eval_in_path_Haze, val_gt_path=args.eval_gt_path_Haze)
    eval_loader_L = get_eval_data(val_in_path=args.eval_in_path_L, val_gt_path=args.eval_gt_path_L)
    eval_loader_Rain = get_eval_data(val_in_path=args.eval_in_path_Rain, val_gt_path=args.eval_gt_path_Rain)
    eval_loader_RealRain = get_eval_data(val_in_path=args.eval_in_path_realRain, val_gt_path=args.eval_in_path_realRain)
    eval_loader_RealSnow = get_eval_data(val_in_path=args.eval_in_path_realSnow, val_gt_path=args.eval_in_path_realSnow)
    eval_loader_RealHaze = get_eval_H_data(val_in_path=args.eval_in_path_realHaze, val_gt_path=args.eval_in_path_realHaze)


    #Derain 
    test(net=net, eval_loader = eval_loader_Rain,  Dname= 'SPA+',flag = [0,1,0],model_flag= args.flag)
    #test(net=net, eval_loader = eval_loader_RealRain, epoch = epoch,  Dname= 'RealRain-fromInterNet',flag = [0,1,0],model_flag= args.flag)
    # Dehaze
    test(net=net, eval_loader = eval_loader_Haze, Dname= 'RealHaze',flag =  [1,0,0],model_flag= args.flag)
    #test(net=net, eval_loader = eval_loader_RealHaze, epoch=epoch, Dname= 'Real_Haze-fromInterNet',flag = [1,0,0],model_flag= args.flag)
    # Desnow
    test(net=net, eval_loader = eval_loader_L, Dname= 'RealSnow',flag =  [0,0,1] ,model_flag= args.flag)
    #test(net=net, eval_loader = eval_loader_RealSnow, epoch = epoch,  Dname= 'RealSnow-fromInterNet',flag = [0,0,1] ,model_flag= args.flag)
    
