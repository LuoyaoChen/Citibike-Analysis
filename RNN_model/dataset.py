import pandas as pd
import numpy as np
import os
import torch 
from torch.utils.data import Dataset
import operator

class CityBikeDataset(Dataset):
   def __init__(self, datafolder_dir):
      self.datafolder_dir = datafolder_dir
      self.months_dir = [os.path.join(self.datafolder_dir, i) for i in os.listdir(self.datafolder_dir)] # 2021-1
      # print("self.months_dir: ", self.months_dir) # ['/mnt/NAS/home/Luoyao/citybike/datafolder/2021-1', ....]
      
      self.days_dir=[]  # ['/mnt/NAS/home/Luoyao/citybike/datafolder/2021-1/2021-1-1', ...]
      for m in self.months_dir:
         # print("m is: ", m)
         # print("m len: ", len(os.listdir(m)))

         for d in os.listdir(m):
            self.days_dir.append(os.path.join(m,d))
                        
      # print("self.days_dir: ", self.days_dir)

       
   def __len__(self):
      return len(self.days_dir) # around 365
    
    
   def __getitem__(self, idx): # input is one day 
      imgs = [] # a list-size-12 of img-size-6*6
      gts = [] # a list-size-11 of list-sized-6
      for hr in os.listdir(self.days_dir[idx]): # hr.csv
         # print("os.path.join(self.days_dir[idx], hr): ", os.path.join(self.days_dir[idx], hr))
         df = pd.read_csv(os.path.join(self.days_dir[idx], hr), index_col =0, header =0)
         # print("df.shape: ", df.values.shape) # df.shape:  (6, 6)
         img = torch.tensor(df.values).unsqueeze(0) # batch
         # print("img.shape: ", img.shape) # img.shape:  torch.Size([1, 6, 6])
         imgs.append(img)
      
      
      ## construct a vector per csv, vector size = [6,1]
      _, H, W = imgs[0].shape
      dist_lis=[]
      for idx, A in enumerate(imgs):
         # print("A.shape: ", A.shape) # A.shape:  torch.Size([1, 6, 6])
         lis = []
         for i in range(H): # distance = sum(col) - sum(row)
            lis.append(torch.sum(A[:,:,i])-torch.sum(A[:,i,:]))
         dist_lis.append(lis)
      for i in range(1,len(dist_lis)):
         # gts.append([(a - b) for a, b in zip(dist_lis[i], dist_lis[i-1])])
         temp =[]
         for a, b in zip(dist_lis[i], dist_lis[i-1]):
            temp.append(a-b)
         temp = [i.unsqueeze(dim=0).unsqueeze(dim=0) for i in temp]
         temp = torch.cat(temp, dim=1)
         # print("temp.shape: ", temp.shape) # temp.shape:  torch.Size([1, 6])
         gts.append(temp)
         
      ## concat to get one day's 12 csvs
      res = torch.concat(imgs[:-1], dim=0) 
      gts = torch.cat([i for i in gts], dim =0)
      # print("res.shape,  gts.shape: ", res.shape,  gts.shape) # res.shape,  gts.shape:  torch.Size([11, 6, 6]) torch.Size([11, 6])
      return {"imgs":res.float(), "gts":gts.float()}
   

if __name__ =="__main__":
   data=CityBikeDataset('/mnt/NAS/home/Luoyao/citibike/datafolder')
   print(len(data))
   print("data[2]['imgs'].shape, len(data[2]['gts']: ", data[2]['imgs'].shape, len(data[2]['gts'])) # data[2]['imgs'].shape, len(data[2]['gts']:  torch.Size([11, 6, 6]) 11
 