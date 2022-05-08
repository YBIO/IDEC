# import os
# import pandas as pd
# from tqdm import tqdm

# # path = r'tsne/feature_tsne/'
# path = r'tsne/label_tsne/'
# save_path = r'tsne/'

# file_list = []
# for i, file in  enumerate(tqdm(os.listdir(path))):
#     df = pd.read_csv(path + file)
#     file_list.append(df)
#     if i>100:
#         break

# result = pd.concat(file_list)   # 合并文件
# print('Done')
# # result.to_csv(save_path + 'feature_tsne.csv', index=False)  
# result.to_csv(save_path + 'label_tsne.csv', index=False)  
# print('Finished')



#-*-coding:utf-8-*-
import os
import pandas as pd
import glob

path = r'tsne/feature_tsne_voc_sample/17_15-1'
# path = r'tsne/label_tsne_voc_sample/17_15-1'

csv_list = glob.glob(os.path.join(path,'*.csv')) #查看同文件夹下的csv文件数
print(u'共发现%s个CSV文件'% len(csv_list))
print(u'Processing............')
count=0
for i in csv_list: #循环读取同文件夹下的csv文件
    count+=1
    fr = open(i,'rb').read()
    with open('tsne/feature_tsne_17_15-1.csv','ab') as f:
    # with open('tsne/label_tsne_17_15-1.csv','ab') as f:
        f.write(fr)
print('合并完毕！')
