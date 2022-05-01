import torch
import os

#dataset
from dataset import CityBikeDataset
from torch.utils.data import DataLoader

# train
from tqdm import tqdm
from torch.nn import DataParallel
from torch.nn import MSELoss
# from model import CityBike_Model
from model_RNN_only import CityBike_Model
from torch import optim

# tensorboard
from torch.utils.tensorboard import SummaryWriter
import shutil

def train(device,device_ids, 
         datafolder_dir, lr, batch_size, max_iter, 
         exp_dir, model_output_dir,tensorboard_dir):
   print(f"****training arguments\n  ****datafolder_dir: {datafolder_dir}\n  ****batch_size: {batch_size}\n  ****model_output_dir: {model_output_dir}")
   assert os.path.isdir(model_output_dir)
   writer = SummaryWriter(tensorboard_dir)

   ## DATASET
   train_data = CityBikeDataset(datafolder_dir)
   print(f"Dataset created, len = {len(train_data)} ....")
   
   ## DATALODER
   train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=False)
   print(f"train_loader created, len = {len(train_loader)} ....")
   
   ## MODEL
   model = CityBike_Model(device=device, 
                          gcn_in_dim =6, gcn_hid_dim=6, gcn_out_dim=6, gcn_dropout=0.1,
                          rnn_in_features=36, rnn_hidden_features=36, rnn_out_features=6, 
                          final_outfeatures = 6)
   model = DataParallel(model, device_ids=device_ids)
   # model = model.to(device)

   model.train()
   
   ## OPTIMIER ect
   optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
   
   ## EPOCH
   iter = 0
   EPOCH = max_iter//batch_size + 1
   avgs = 0.0 # epoch average
   sum = 0.0
   for epoch in tqdm(range(EPOCH)):
      for idx, batch in enumerate(train_loader):
         # print("batch.keys: ", batch.keys)
         one_day = batch['imgs'].cuda(device)
         labels = batch['gts'].cuda(device)
         # print("one_day.shape, labels.shape: ", one_day.shape, labels.shape)  # one_day.shape, labels.shape:  torch.Size([32, 11, 6, 6]) torch.Size([32, 11, 6])
         
         ## FIT MODEL
         out = model(one_day)
         # print("out.shape : ", out.shape) # out.shape :  torch.Size([32, 11, 6])
         
         ## LOSS 
         optimizer.zero_grad()
         criterion= MSELoss()
         loss = criterion(out, labels)
         # print("out.dtype, labels.dtype, loss: ", out.dtype, labels.dtype,loss) # out.dtype, labels.dtype, loss:  torch.float32 torch.float64 tensor(76.5080, device='cuda:1', dtype=torch.float64, grad_fn=<MseLossBackward0>)
         loss.backward()
         optimizer.step()
         
         # SAVE MODEL, LOSS
         if iter % 10 ==0:
            print(f"iter = {iter}, epoch: {epoch}, loss: {loss.item()}")
            writer.add_scalar('loss(MSE)', loss, iter)
         
         if iter % 1000 ==0:
            print(f"saving model CityBike_Model_iter_{iter}.pt to dir: {model_output_dir}...")
            torch.save(model.module.state_dict(), os.path.join(model_output_dir, f'CityBike_Model_iter_{iter}.pt'))
            # torch.save(model.stat_dict(), os.path.join(model_output_dir, f'CityBike_Model_iter{iter}.pt'))
         
         iter +=1
         sum += loss.detach()
   
   print(f"saving model CityBike_Model_iter_{iter}.pt to dir: {model_output_dir}...")
   print(f"sum = {sum}, iter= {iter}, average MSE = {sum / (iter)}")
   torch.save(model.module.state_dict(), os.path.join(model_output_dir, f'CityBike_Model_iter_{iter}.pt'))
   

if __name__ == "__main__":
#  os.environ["CUDA_VISIBLE_DEVICES"]="0"
   note = 'month_3_after'
   device_ids = [1,2]
   device = torch.device('cuda', device_ids[0])

   # seeds
   torch.manual_seed(3001)

   # dataset pram
   datafolder_dir = '/mnt/NAS/home/Luoyao/citibike/datafolder'
   
   # train param
   batch_size = 32
   # EPOCH = 100
   lr = 1e-4
   max_iter= 3000
   
   # exp dir
   exp_dir = f'/mnt/NAS/home/Luoyao/citibike/experiments'

   ## code dir
   if not os.path.exists(os.path.join(exp_dir)):
      os.makedirs(exp_dir)
   if os.path.exists(os.path.join(exp_dir, note,'code')):
      temp=os.path.join(exp_dir, note,'code')
      shutil.rmtree(temp)
   print("shutil.rmtree(os.path.join(exp_dir, note, 'code') finished...")

   from_dir = os.getcwd()
   destination = shutil.copytree(from_dir, exp_dir + '/'+ note +'/' +'code',
                     ignore = shutil.ignore_patterns('__pycache__', '.git', 'hour_csv*', 'datafolder*', 'experiments'))
   print("shutil.copytree finished...")
   
   ## model_output_folder
   model_output_dir = os.path.join(exp_dir, note, 'trains')
   if not os.path.exists(model_output_dir):
      os.makedirs(model_output_dir)
   
   ## tensorboad
   tensorboard_dir = os.path.join(exp_dir, note, "tensorboard_runs") #'/mnt/NAS/home/Luoyao/citibike/runs/trains'
      
   
   print(f"running exp {exp_dir}")
   train(device,device_ids, 
         datafolder_dir, lr, batch_size, max_iter, 
         exp_dir, model_output_dir,tensorboard_dir)