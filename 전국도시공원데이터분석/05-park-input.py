#!/usr/bin/env python
# coding: utf-8

# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://bit.ly/open-data-05-park-input)
# 
# 
# # 전국 도시 공원 표준 데이터
# https://www.data.go.kr/dataset/15012890/standard.do
# * 전국 도시 공원 표준데이터에는 데이터를 전처리 해보기에 좋은 데이터가 많습니다.
# * 시간, 결측치, 이상치, 수치형, 범주형 데이터를 고르게 볼 수 있으며 다양한 텍스트 데이터 처리를 해볼 수 있습니다.
# * 또 정규표현식을 활용해서 텍스트 데이터 전처리와 데이터 마스킹 기법에 대해 다룹니다.
# * 그리고 이렇게 전처리한 내용을 바탕으로 전국 도시공원에 대한 분포를 시각화해 봅니다.
# * 어떤 공원이 어느 지역에 어떻게 분포되어 있는지를 위경도로 표현해 봅니다.
# 
# ## 이번 챕터에서 설치가 필요한 도구
# 
# * 별도의 설치가 필요합니다.(folium 을 설치했던 것 처럼 따로 설치해야지만 사용할 수 있습니다.)
# 
# * 윈도우
#     * <font color="red">주피터 노트북 상에서 설치가 되지 않으니</font> anaconda prompt 를 열어서 설치해 주세요.
#     * <font color="red">관리자 권한</font>으로 아나콘다를 설치하셨다면 다음의 방법으로 anaconda prompt 를 열어 주세요.
#     <img src="https://i.imgur.com/GhoLwsd.png">
# * 맥
#     * terminal 프로그램을 열어 설치해 주세요. 
# 
# 
# ### Pandas Profiling
# * [pandas-profiling/pandas-profiling: Create HTML profiling reports from pandas DataFrame objects](https://github.com/pandas-profiling/pandas-profiling)
# 
# * 2020년 4월 기준 판다스 1.0 이상 버전을 지원하지 않습니다.
# * 아나콘다로 주피터를 설치했다면 : `conda install -c conda-forge pandas-profiling`
# * pip로 주피터를 설치했다면 : `pip install pandas-profiling`
# 
# ### 워드클라우드
# [amueller/word_cloud: A little word cloud generator in Python](https://github.com/amueller/word_cloud)
# 
# * 다음 명령어로 설치가 가능합니다. conda prompt 혹은 터미널을 열어 설치해 주세요.
# 
# * conda : `conda install -c conda-forge wordcloud`
# * pip : `pip install wordcloud`
# 

# ## 분석에 사용할 도구를 불러옵니다.

# In[1]:


# 필요한 라이브러리를 로드합니다.
# pandas, numpy, seaborn, matplotlib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# ## Seaborn 설정으로 시각화의 스타일, 폰트 설정하기
# * [matplotlib.pyplot.rc — Matplotlib 3.1.3 documentation](https://matplotlib.org/3.1.3/api/_as_gen/matplotlib.pyplot.rc.html)

# In[2]:


# seaborn 의 set 기능을 통해 폰트, 마이너스 폰트 설정, 스타일 설정을 합니다.
# Wind : "Malgun Gothic", MAC:"AppleGothic"
sns.set(font="Malgun Gothic", style="darkgrid", rc={"axes.unicode_minus":False})


# In[3]:


# 한글폰트 설정 확인을 합니다.
pd.Series([1, -1, 0, 5, -5]).plot(title="한글폰트")


# In[4]:


# 그래프가 선명하게 표시되도록 합니다.
from IPython.display import set_matplotlib_formats
set_matplotlib_formats("retina")


# ## 데이터 로드

# In[5]:


# 데이터를 로드해서 df 라는 변수에 담습니다.
df = pd.read_csv("C:/Users/alstj/전국도시공원표준데이터/전국도시공원표준데이터.csv", encoding="cp949")
df.shape


# In[6]:


# 미리보기를 합니다.
df.head()


# ## Pandas Profiling
# * [pandas-profiling/pandas-profiling: Create HTML profiling reports from pandas DataFrame objects](https://github.com/pandas-profiling/pandas-profiling)
# 
# * 별도의 설치가 필요합니다.(folium 을 설치했던 것 처럼 따로 설치해야지만 사용할 수 있습니다.)
# * conda : `conda install -c conda-forge pandas-profiling`
# * pip : `pip install pandas-profiling`

# In[7]:


import pandas_profiling


# * 미리 생성해 놓은 리포트 보기 : https://corazzon.github.io/open-data-analysis-basic/05-park_pandas_profile.html

# In[8]:


# pandas_profiling 의 ProfileReport 를 불러와 표현합니다.
# 이 때 title은 "도시공원 표준 데이터" 로 하고 주피터 노트북에서 바로 보면 iframe을 통해 화면이 작게 보이기 때문에
# 별도의 html 파일로 생성해서 그려보세요.
from pandas_profiling import ProfileReport

profile = ProfileReport(df, title="도시공원 분석 데이터")
profile.to_file(output_file="05-park_pandas_profile.html")


# ## 기본 정보 보기

# In[9]:


# info로 기본 정보를 봅니다.
df.info()


# In[10]:


# 결측치의 수를 구합니다.
df.isnull().sum()


# In[11]:


# 결측치 비율 구하기
# 결측의 평균을 통해 비율을 구하고 100을 곱해줍니다.
round(df.isnull().mean() * 100, 2)


# ## 결측치 시각화
# * [ResidentMario/missingno: Missing data visualization module for Python.](https://github.com/ResidentMario/missingno)

# In[12]:


# 폰트 설정이 해제되었다면 다시 설정해 주세요.
sns.set(font="Malgun Gothic", style="darkgrid", rc={"axes.unicode_minus":False})


# In[13]:


# 이전 챕터에서 설치하지 않았다면 아나콘다에 missingno를 설치합니다. 
# !conda install -c conda-forge missingno
# 라이브러리를 로드합니다.
import missingno

missingno.matrix(df)


# * 그래프의 색상 선택 : [Choosing Colormaps in Matplotlib — Matplotlib 3.1.0 documentation](https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html)

# In[14]:


# seaborn을 통해 위와 같은 결측치 그래프를 그려봅니다.
null = df.isnull()
plt.figure(figsize=(15, 4))
sns.heatmap(null, cmap="Blues_r")


# # 데이터 전처리
# * 불필요한 컬럼 제거
# * 시도 결측치 처리
#     * 다른 데이터로 대체
#     * 도로명 혹은 지번 둘 중 하나만 있다면 있는 데이터로 대체
# * 아웃라이어 데이터 제거 혹은 대체
#     * 위경도가 국내 범위를 벗어나는 데이터의 경우 제외하고 그리도록 처리

# ## 불필요한 컬럼 제거하기

# In[15]:


# 전체 컬럼명을 출력해 봅니다.
df.columns


# In[16]:


# drop 으로 'Unnamed_19' 를 제거하기
print(df.shape)
df = df.drop(['Unnamed_19'], axis=1)
print(df.shape)


# ## 결측치 대체
# ### 도로명 주소와 지번 주소 
# * 둘 중 하나만 있을 때 나머지 데이터로 결측치 대체하기

# In[17]:


# 도로명 주소의 널값 수
df["소재지도로명주소"].isnull().sum()


# In[18]:


# 지번 주소의 널값 수
df["소재지지번주소"].isnull().sum()


# In[19]:


# "소재지도로명주소"와 "소재지지번주소"가 모두 결측치가 아닌 데이터를 찾습니다.
df[df["소재지도로명주소"].notnull() & df["소재지지번주소"].notnull()].shape


# In[20]:


# "소재지도로명주소"의 결측치를 fillna 를 통해 "소재지지번주소"로 채웁니다.
df["소재지도로명주소"] = df["소재지도로명주소"].fillna(df["소재지지번주소"])


# In[21]:


# "소재지도로명주소"의 결측치수를 세어봅니다.
df["소재지도로명주소"].isnull().sum()


# In[22]:


# "소재지도로명주소"와 "소재지지번주소"가 모두 결측치인 데이터를 찾습니다.
df[df["소재지도로명주소"].isnull() & df["소재지지번주소"].isnull()].shape


# ## 파생변수 만들기
# ### 주소를 통한 시도, 구군 변수 생성하기

# In[23]:


# 소재지도로명주소로 시도, 구군 변수 생성하기
# .str.split(' ', expand=True)[0] 을 통해 공백문자로 분리하고 리스트의 첫번째 값을 가져오도록 하기
df["시도"] = df["소재지도로명주소"].str.split(expand=True)[0]
df[["소재지도로명주소", "시도"]].head(3)


# In[24]:


# 구군 가져오기
df["구군"] = df["소재지도로명주소"].str.split(expand=True)[1]
df[["소재지도로명주소", "시도", "구군"]].sample(3)


# In[25]:


# 시도 데이터의 빈도수 세어보기
df["시도"].value_counts()


# In[26]:


# 강원은 "강원도"로 변경해줄 필요가 보입니다.
df["시도"] = df["시도"].replace("강원","강원도")
df["시도"].value_counts()


# ## 이상치 제거
# * 경도, 위도의 이상치 처리하기

# In[27]:


# 위경도 시각화
sns.scatterplot(data=df, x="경도", y="위도")


# In[28]:


# 위 지도로 위도와 경도의 아웃라이어 데이터를 제외하고 출력해 봅니다.
# 좀 더 정확하게 출력하려면 대한민국 위경도 데이터 범위를 다시 넣어줍니다. 
# 이상치를 제거한 데이터를 df_park 라는 새로운 변수에 담습니다.
df_park = df[(df["경도"] < 132) & (df["위도"] > 32)].copy()


# In[29]:


# 위도 경도의 아웃라이어 데이터가 제거되었는지 확인함
sns.scatterplot(data=df_park, x="경도", y="위도")


# In[30]:


# 위도와 경도의 요약값을 describe 로 봅니다.
df[["위도", "경도"]].describe()


# In[31]:


# 위경도가 잘못입력된 데이터를 봅니다.
# 주소가 잘못되지는 않았습니다.
# 주소를 통해 위경도를 다시 받아올 필요가 있습니다.
df[(df["경도"] > 132) | (df["위도"] < 32)]