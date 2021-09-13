#!/usr/bin/env python
# coding: utf-8

# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://bit.ly/open-data-01-apt-price-input)
# 
# # 전국 신규 민간 아파트 분양가격 동향
# 
# 2013년부터 최근까지 부동산 가격 변동 추세가 아파트 분양가에도 반영될까요? 공공데이터 포털에 있는 데이터를 Pandas 의 melt, concat, pivot, transpose 와 같은 reshape 기능을 활용해 분석해 봅니다. 그리고 groupby, pivot_table, info, describe, value_counts 등을 통한 데이터 요약과 분석을 해봅니다. 이를 통해 전혀 다른 형태의 두 데이터를 가져와 정제하고 병합하는 과정을 다루는 방법을 알게 됩니다. 전처리 한 결과에 대해 수치형, 범주형 데이터의 차이를 이해하고 다양한 그래프로 시각화를 할 수 있게 됩니다.
# 
# 
# ## 다루는 내용
# * 공공데이터를 활용해 전혀 다른 두 개의 데이터를 가져와서 전처리 하고 병합하기
# * 수치형 데이터와 범주형 데이터를 바라보는 시각을 기르기
# * 데이터의 형식에 따른 다양한 시각화 방법 이해하기
# 
# ## 실습
# * 공공데이터 다운로드 후 주피터 노트북으로 로드하기
# * 판다스를 통해 데이터를 요약하고 분석하기
# * 데이터 전처리와 병합하기
# * 수치형 데이터와 범주형 데이터 다루기
# * 막대그래프(bar plot), 선그래프(line plot), 산포도(scatter plot), 상관관계(lm plot), 히트맵, 상자수염그림, swarm plot, 도수분포표, 히스토그램(distplot) 실습하기
# 
# ## 데이터셋
# * 다운로드 위치 : https://www.data.go.kr/dataset/3035522/fileData.do
# 
# ### 전국 평균 분양가격(2013년 9월부터 2015년 8월까지)
# * 전국 공동주택의 3.3제곱미터당 평균분양가격 데이터를 제공
# 
# ###  주택도시보증공사_전국 평균 분양가격(2019년 12월)
# * 전국 공동주택의 연도별, 월별, 전용면적별 제곱미터당 평균분양가격 데이터를 제공
# * 지역별 평균값은 단순 산술평균값이 아닌 가중평균값임

# In[33]:


# 파이썬에서 쓸 수 있는 엑셀과도 유사한 판다스 라이브러리를 불러옵니다.
import pandas as pd


# ## 데이터 로드
# ### 최근 파일 로드
# 공공데이터 포털에서 "주택도시보증공사_전국 평균 분양가격"파일을 다운로드 받아 불러옵니다.
# 이 때, 인코딩을 설정을 해주어야 한글이 깨지지 않습니다.
# 보통 엑셀로 저장된 한글의 인코딩은 cp949 혹은 euc-kr로 되어 있습니다.
# df_last 라는 변수에 최근 분양가 파일을 다운로드 받아 로드합니다.
# 
# * 한글인코딩 : [‘설믜를 설믜라 못 부르는’ 김설믜씨 “제 이름을 지켜주세요” : 사회일반 : 사회 : 뉴스 : 한겨레](http://www.hani.co.kr/arti/society/society_general/864914.html)
# 
# 데이터를 로드한 뒤 shape를 통해 행과 열의 갯수를 출력합니다.

# In[3]:


# 최근 분양가 파일을 로드해서 df_last 라는 변수에 담습니다.
# 파일로드시 OSError가 발생한다면, engine="python"을 추가해 보세요.
# 윈도우에서 파일탐색기의 경로를 복사해서 붙여넣기 했는데도 파일을 불러올 수 없다면
# 아마도 경로에 있는 ₩ 역슬래시 표시를 못 읽어왔을 가능성이 큽니다. 
# r"경로명" 으로 적어주세요.
# r"경로명"으로 적게 되면 경로를 문자 그대로(raw) 읽으라는 의미입니다.
df_last = pd.read_csv(r"C:/Users/alstj/전국신규민간아파트분양가격동향/주택도시보증공사_전국 평균 분양가격(2019년 12월).csv", encoding="cp949")
df_last.shape


# In[4]:


# head 로 파일을 미리보기 합니다.
# 메소드 뒤에 ?를 하면 자기호출 이라는 기능을 통해 메소드의 docstring을 출력합니다.
# 메소드의 ()괄호 안에서 Shift + Tab키를 눌러도 같은 문서를 열어볼 수 있습니다.
# Shift + Tab + Tab 을 하게 되면 팝업창을 키울 수 있습니다.
df_last.head()


# In[5]:


# tail 로도 미리보기를 합니다.
df_last.tail()


# ### 2015년 부터 최근까지의 데이터 로드
# 전국 평균 분양가격(2013년 9월부터 2015년 8월까지) 파일을 불러옵니다.
# df_first 라는 변수에 담고 shape로 행과 열의 갯수를 출력합니다.

# In[6]:


# 해당되는 폴더 혹은 경로의 파일 목록을 출력해 줍니다.
get_ipython().run_line_magic('ls', '"C:/Users/alstj/전국신규민간아파트분양가격동향"')


# In[7]:


# df_first 에 담고 shape로 행과 열의 수를 출력해 봅니다.
df_first = pd.read_csv("C:/Users/alstj/전국신규민간아파트분양가격동향/전국 평균 분양가격(2013년 9월부터 2015년 8월까지).csv", encoding="cp949")
df_first.shape


# In[8]:


# df_first 변수에 담긴 데이터프레임을 head로 미리보기 합니다.
df_first.head()


# In[9]:


# df_first 변수에 담긴 데이터프레임을 tail로 미리보기 합니다.
df_first.tail()


# ### 데이터 요약하기

# In[10]:


# info 로 요약합니다.
df_last.info()


# ### 결측치 보기

# isnull 혹은 isna 를 통해 데이터가 비어있는지를 확인할 수 있습니다.
# 결측치는 True로 표시되는데, True == 1 이기 때문에 이 값을 다 더해주면 결측치의 수가 됩니다.

# In[11]:


# isnull 을 통해 결측치를 봅니다.
df_last.isnull()


# In[12]:


# isnull 을 통해 결측치를 구합니다.
df_last.isnull().sum()


# In[13]:


# isna 를 통해 결측치를 구합니다.
df_last.isna().sum()


# ### 데이터 타입 변경
# 분양가격이 object(문자) 타입으로 되어 있습니다. 문자열 타입을 계산할 수 없기 때문에 수치 데이터로 변경해 줍니다. 결측치가 섞여 있을 때 변환이 제대로 되지 않습니다. 그래서 pd.to_numeric 을 통해 데이터의 타입을 변경합니다.

# In[14]:


type(np.nan)


# In[15]:


df_last["분양가격"] = pd.to_numeric(df_last["분양가격(㎡)"], errors="coerce")
df_last["분양가격"].head(1)


# ### 평당분양가격 구하기
# 공공데이터포털에 올라와 있는 2013년부터의 데이터는 평당분양가격 기준으로 되어 있습니다.
# 분양가격을 평당기준으로 보기위해 3.3을 곱해서 "평당분양가격" 컬럼을 만들어 추가해 줍니다.

# In[16]:


df_last["평당분양가격"] = df_last["분양가격"] * 3.3
df_last.head(1)


# ### 분양가격 요약하기

# In[17]:


# info를 통해 분양가격을 봅니다.
df_last.info()


# In[18]:


# 변경 전 컬럼인 분양가격(㎡) 컬럼을 요약합니다.
df_last["분양가격(㎡)"].describe()


# In[19]:


# 규모구분의 unique 값 보기
df_last["규모구분"].unique()


# In[20]:


# 규모구분을 전용면적으로 변경하기
df_last["전용면적"] = df_last["규모구분"].str.replace("전용면적", "")
df_last["전용면적"] = df_last["전용면적"].str.replace("초과", "~")
df_last["전용면적"] = df_last["전용면적"].str.replace("이하", "")
df_last["전용면적"] = df_last["전용면적"].str.replace(" ", "").str.strip()
df_last["전용면적"]


# ### 필요없는 컬럼 제거하기
# drop을 통해 전처리 해준 컬럼을 제거합니다. pandas의 데이터프레임과 관련된 메소드에는 axis 옵션이 필요할 때가 있는데 행과 열중 어떤 기준으로 처리를 할 것인지를 의미합니다. 보통 기본적으로 0으로 되어 있고 행을 기준으로 처리함을 의미합니다. 메모리 사용량이 줄어들었는지 확인합니다.

# In[21]:


# info로 정보 보기
df_last.info()


# In[22]:


# drop 사용시 axis에 유의 합니다.
# axis 0:행, 1:열
df_last = df_last.drop(["규모구분", "분양가격(㎡)"], axis=1)


# In[23]:


# 제거가 잘 되었는지 확인 합니다.
df_last.head(1)


# In[24]:


# 컬럼 제거를 통해 메모리 사용량이 줄어들었는지 확인합니다.
df_last.info()


# ## groupby 로 데이터 집계하기
# groupby 를 통해 데이터를 그룹화해서 연산을 해봅니다.

# In[39]:


# 지역명으로 분양가격의 평균을 구하고 막대그래프(bar)로 시각화 합니다.
# df.groupby(["인덱스로 사용할 컬럼명"])["계산할 컬럼 값"].연산()
df_last.groupby(["지역명"])["평당분양가격"].mean()


# In[40]:


# 전용면적으로 분양가격의 평균을 구합니다.
df_last.groupby(["전용면적"])["평당분양가격"].mean()


# In[43]:


# 지역명, 전용면적으로 평당분양가격의 평균을 구합니다.
df_last.groupby(["지역명", "전용면적"])["평당분양가격"].mean().unstack().round()


# In[49]:


# 연도, 지역명으로 평당분양가격의 평균을 구합니다.
g = df_last.groupby(["연도", "지역명"])["평당분양가격"].mean()
g
#g.unstack().transpose()


# ## pivot table 로 데이터 집계하기
# * groupby 로 했던 작업을 pivot_table로 똑같이 해봅니다.

# In[50]:


# 지역명을 index 로 평당분양가격 을 values 로 구합니다.
pd.pivot_table(df_last, index=["지역명"], values=["평당분양가격"], aggfunc="mean")


# In[52]:


# df_last.groupby(["전용면적"])["평당분양가격"].mean()


# In[56]:


# 전용면적을 index 로 평당분양가격 을 values 로 구합니다.
pd.pivot_table(df_last, index=["전용면적"], values=["평당분양가격"])


# In[ ]:


# 지역명, 전용면적으로 평당분양가격의 평균을 구합니다.
# df_last.groupby(["전용면적", "지역명"])["평당분양가격"].mean().unstack().round()


# In[58]:


pd.pivot_table(df_last, index=["전용면적"], columns=["지역명"], values=["평당분양가격"]).round()


# In[59]:


# 연도, 지역명으로 평당분양가격의 평균을 구합니다.
# g = df_last.groupby(["연도", "지역명"])["평당분양가격"].mean()
p = pd.pivot_table(df_last, index=["연도", "지역명"], values=["평당분양가격"])
p.loc[2017]


# ## 최근 데이터 시각화 하기
# ### 데이터시각화를 위한 폰트설정
# 한글폰트 사용을 위해 matplotlib의 pyplot을 plt라는 별칭으로 불러옵니다.

# In[62]:


import matplotlib.pyplot as plt

plt.rc("font", family="Malgun Gothic")


# ### Pandas로 시각화 하기 - 선그래프와 막대그래프
# pandas의 plot을 활용하면 다양한 그래프를 그릴 수 있습니다.
# seaborn을 사용했을 때보다 pandas를 사용해서 시각화를 할 때의 장점은 미리 계산을 하고 그리기 때문에 속도가 좀 더 빠릅니다.

# In[67]:


# 지역명으로 분양가격의 평균을 구하고 선그래프로 시각화 합니다.
g = df_last.groupby(["지역명"])["평당분양가격"].mean().sort_values(ascending=False)
g.plot()


# In[68]:


# 지역명으로 분양가격의 평균을 구하고 막대그래프(bar)로 시각화 합니다.
# g.plot(kind="bar")
g.plot.bar(rot=0, figsize=(10, 3))


# 전용면적별 분양가격의 평균값을 구하고 그래프로 그려봅니다.

# In[69]:


# 전용면적으로 분양가격의 평균을 구하고 막대그래프(bar)로 시각화 합니다.
df_last.groupby(["전용면적"])["평당분양가격"].mean().plot.bar()


# In[70]:


# 연도별 분양가격의 평균을 구하고 막대그래프(bar)로 시각화 합니다.
df_last.groupby(["연도"])["평당분양가격"].mean().plot.bar()


# ### box-and-whisker plot | diagram
# 
# * https://pandas.pydata.org/pandas-docs/stable/user_guide/visualization.html
# * https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.boxplot.html
# 
# * [상자 수염 그림 - 위키백과, 우리 모두의 백과사전](https://ko.wikipedia.org/wiki/%EC%83%81%EC%9E%90_%EC%88%98%EC%97%BC_%EA%B7%B8%EB%A6%BC)
# * 가공하지 않은 자료 그대로를 이용하여 그린 것이 아니라, 자료로부터 얻어낸 통계량인 5가지 요약 수치로 그린다.
# * 5가지 요약 수치란 기술통계학에서 자료의 정보를 알려주는 아래의 다섯 가지 수치를 의미한다.
# 
# 
# 1. 최솟값
# 1. 제 1사분위수
# 1. 제 2사분위수( ), 즉 중앙값
# 1. 제 3 사분위 수( )
# 1. 최댓값
# 
# * Box plot 이해하기 : 
#     * [박스 플롯에 대하여 :: -[|]- Box and Whisker](https://boxnwhis.kr/2019/02/19/boxplot.html)
#     * [Understanding Boxplots – Towards Data Science](https://towardsdatascience.com/understanding-boxplots-5e2df7bcbd51)

# In[74]:


# index를 월, columns 를 연도로 구하고 평당분양가격 으로 pivot_table 을 구하고 상자수염그림을 그립니다.
df_last.pivot_table(index="월", columns="연도", values="평당분양가격").plot.box()


# In[76]:


# columns 에 "연도", "전용면적"을 추가해서 pivot_table 을 만들고 시각화 합니다.
p = df_last.pivot_table(index="월", columns=["연도", "전용면적"], values="평당분양가격")
p.plot.box(figsize=(15, 3), rot=30)


# In[80]:


# index를 월, columns 를 지역명으로 구하고 평당분양가격 으로 pivot_table 을 구하고 선그래프를 그립니다.
p = df_last.pivot_table(index="월", columns="지역명", values="평당분양가격")
p.plot(figsize=(15, 3), rot=30)