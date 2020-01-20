#!/usr/bin/env python
# coding: utf-8

# In[36]:


import cv2  #이미지 처리 (open cv)
from matplotlib import pyplot as plt # 그래프 그리기
import numpy as np # 배열 연산
from sklearn.model_selection import train_test_split # 머신러닝 패키지
#뭉쳐있는 데이터셋을 트레이닝셋과 테스트셋으로 분리해주는 아주 편리한 함수
import pandas as pd # 데이터프레임(엑셀과 비슷)

#-------------------딥러닝 패키지--------------------------------------------------
from keras.layers import Input, Activation, Conv2D, Flatten, Dense, MaxPooling2D, Dropout, Add, LeakyReLU
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
#-------------------딥러닝 패키지-----------------------------------------
import datetime
import os, sys, glob
plt.style.use('dark_backgr')


# In[77]:


get_ipython().run_line_magic('ls', '')


# # Read and Analyze Labels
# 

# In[83]:


get_ipython().run_line_magic('ls', '/Users/youngsik_won/Downloads/train_test_files/All_labels.txt')


# In[90]:


labels = pd.read_csv("/Users/youngsik_won/Downloads/train_test_files/All_labels.txt"
                    ,sep=" ", header=None)
# pd.read_Csv() 텍스트 파일을 읽어 데이터프레임 형태로 만든다
labels.head() #데이터프레임의 첫 5줄을 출력한다




# In[93]:


labels.describe #데이터셋의 통계적인 개요를 출력한다


# 뷰티스코어의 평균은 2.99점이다.

# 표준편차는  0.68

# 최소값은 1.01


# 최대값은 4.75


# #Dataframe to Numpy Array

# In[96]:


labels_np = labels.values # .values  데이터 프레임을 numpy 배열로 변환한다.
print(lables_np[:5]) 


# # Read Images

# In[103]:


imgs = np.empty((len(labels_np), 350, 350, 3), dtype=np.uint8)
 #np.empty() 지정된 크기로 빈 배열을 만든다.
    
for i, (img_filename, rating) in enumerate(labels_np):
    img = cv2.imread(os.path.join('Images', img_filename))
      # cv2.imread() 이미지 파일을 로드한다.
        
        
    if img.shape[0] != 350 or img.shape[1] != 350:
        print(img_filename)
        # 이미지 파일의 크기가 맞는지 체크 해줌.
        
    imgs[i] = img


# In[ ]:




