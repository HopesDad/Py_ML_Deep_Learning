#!/usr/bin/env python
# coding: utf-8

# In[28]:


#행렬 연산 패키지
import numpy as np
# 데이터 분석 키지
import pandas as pd
import json


# In[29]:


#Load Dataset

meta = pd.read_csv('./input/movies_metadata.csv') # csv 파일 로드할 시 주의할 점!! < 경로를 잘맞춰야한다.>

meta.head() #데이터프레임의 처음 5줄을 출력한다


# In[30]:


meta = meta[['id', 'original_title', 'original_language', 'genres']]
meta = meta.rename(columns={'id':'movieId'})
meta = meta[meta['original_language'] == 'en']
meta.head()


# In[31]:


ratings = pd.read_csv('./input/ratings_small.csv') # csv 파일 로드할 시 주의할 점!! < 경로를 잘맞춰야한다.>
ratings = ratings[['userId', 'movieId', 'rating']]
 # 어떤 유저가 어떤 영화의 index로 점수를 매긴다. 
ratings.head() #데이터프레임의 처음 5줄을 출력한다


# In[32]:


ratings.describe()


# In[33]:


meta.movieId = pd.to_numeric(meta.movieId, errors='coerce')#to_numeric() 문자열을 숫자 타입으로 변환한다.
ratings.movieId = pd.to_numeric(ratings.movieId, errors='coerce')
# meta의 무비 ID와 ratings. 무비 아이디를 숫자로 바꾼다.


# In[34]:


def parse_genres(genres_str):
    genres = json.loads(genres_str.replace('\'', '"')) # 작은 따음표를 큰따옴표로 변경
    
    genres_list = []
    for g in genres:
        genres_list.append(g['name'])
  # String 으로 배열로 저장 
    return genres_list

meta['genres'] = meta['genres'].apply(parse_genres)

meta.head()


# In[35]:


## Merge Meta and Ratings

data = pd.merge(ratings, meta, on='movieId', how='inner')

# 두 개의 데이터 프레임을 병합한다. 

data.head()


# In[36]:


#Pivoy Table

matrix = data.pivot_table(index='userId', columns='original_title', values='rating')
 #df.pivot_table_create pivot_table
matrix.head(20)

# NaN has a low value because the user does not watch the same movie.


# In[39]:


GENRE_WEIGHT = 0.1

def pearsonR(s1, s2):
    s1_c = s1 - s1.mean()
    s2_c = s2 - s2.mean()
    return np.sum(s1_c * s2_c) / np.sqrt(np.sum(s1_c ** 2) * np.sum(s2_c ** 2))

def recommend(input_movie, matrix, n, similar_genre=True):
    input_genres = meta[meta['original_title'] == input_movie]['genres'].iloc(0)[0]

    result = []
    for title in matrix.columns:
        if title == input_movie:
            continue

        # rating comparison
        cor = pearsonR(matrix[input_movie], matrix[title])
        
        # genre comparison
        if similar_genre and len(input_genres) > 0:
            temp_genres = meta[meta['original_title'] == title]['genres'].iloc(0)[0]

            same_count = np.sum(np.isin(input_genres, temp_genres))
            cor += (GENRE_WEIGHT * same_count)
            
            
            
        if np.isnan(cor): # np.isin() Compare the array and return Ture if the same elemnet exists.
            continue
        else:
            result.append((title, '{:.2f}'.format(cor), temp_genres))
            
            
            
    result.sort(key=lambda r: r[1], reverse=True)
     # Rates in high order, reverse descending order
    
    return result[:n] # Return 10 pieces
            
            


# In[41]:


recommend_result = recommend('The Dark Knight', matrix, 10, similar_genre=True)

pd.DataFrame(recommend_result, columns = ['Title', 'Correlation', 'Genre'])


# In[ ]:




