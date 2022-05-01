from gettext import npgettext
import os
import shutil
from tqdm import tqdm


hour_csv_folder = '/mnt/NAS/home/Luoyao/citybike/hour_csv/content/drive/MyDrive/3001_Proj/preprocessed/hour_csv'
datafolder_dir = '/mnt/NAS/home/Luoyao/citybike/datafolder'
for one in os.listdir(hour_csv_folder):
   month = one[0:7] # 2021-02
   date = one[0:10] # 2021-02-10
   hour = one[11:-4] # 0
   # print("month, date, hour: ", month, date, hour)
   target = os.path.join(datafolder_dir, month, date)#, hour)
   # print('hour_csv_folder + one, target: ', hour_csv_folder + one, target)
   os.makedirs(target, exist_ok=True)
   shutil.copyfile(os.path.join(hour_csv_folder, one), os.path.join(target,hour)+'.csv')
