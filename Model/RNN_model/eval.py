import torch
import os

#dataset
from dataset import CityBikeDataset
from torch.utils.data import DataLoader

# train
from tqdm import tqdm
from torch.nn import DataParallel
from torch.nn import MSELoss
from model import CityBike_Model

# tensorboard
from torch.utils.tensorboard import SummaryWriter

def eval(device, datafolder_dir, batch_size, pretrained_model_dir,tensorboard_dir):
   ## DATASET
   eval_data = CityBikeDataset(datafolder_dir)
   print(f"Dataset created, len = {len(eval_data)}!....")
   
   ## DATALODER
   eval_loader = DataLoader(eval_data, batch_size=batch_size, shuffle=False)
   print(f"train_loader created, len = {len(eval_loader)}!....")
   
   ## MODEL
   model = CityBike_Model(device = device, gcn_in_dim =6, gcn_hid_dim=6, gcn_out_dim=6, gcn_dropout=0.1,
                          rnn_in_features=36, rnn_hidden_features=36, rnn_out_features=6).to(device)
   sum_dict = {}
   for model_dir in tqdm(os.listdir(pretrained_model_dir)):
      writer = SummaryWriter(tensorboard_dir)
      model.load_state_dict(torch.load(os.path.join(pretrained_model_dir, model_dir)))
      model.eval()
      print(f"model loaded from {pretrained_model_dir} ...")
      ## EPOCH
      sum = 0.0
      with torch.no_grad():
         for idx, batch in enumerate(eval_loader):
            one_day = batch['imgs'].cuda(device)
            labels = batch['gts'].cuda(device)
            # print("one_day.shape, labels.shape: ", one_day.shape, labels.shape)
            
            ## FIT MODEL
            out = model(one_day)
            print("out.shape : ", out.shape)
            
            ## LOSS 
            criterion= MSELoss()
            loss = criterion(out, labels) 
            sum += loss.detach()
            print(f"idx: {idx}, loss: {loss.item()}")
            writer.add_scalar(f'test_loss(MSE)_model_{model_dir}', loss.detach(), idx)
      print(f"sum = {sum}, iter= {idx+1}, average MSE = {sum / (idx+1)}")
      sum_dict[model_dir]=sum / (idx+1)
   print(sum_dict)
   

if __name__ == "__main__":
#  os.environ["CUDA_VISIBLE_DEVICES"]="0"
 device = torch.device('cuda', 1)
 note = '3k_iter'
 
 tensorboard_dir = f'/mnt/NAS/home/Luoyao/citibike/experiments/{note}/tensorboard_runs'
 # dataset pram
 datafolder_dir = '/mnt/NAS/home/Luoyao/citibike/datafolder'
 
 # train param
 batch_size = 32
 # pretrained model path
 pretrained_model_dir = '/mnt/NAS/home/Luoyao/citibike/experiments/{note}/trains'
 eval(device, datafolder_dir, batch_size, pretrained_model_dir, tensorboard_dir)