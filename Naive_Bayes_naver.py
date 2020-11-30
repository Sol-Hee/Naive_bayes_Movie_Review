
from urllib.request import urlopen
import urllib.request as req
from bs4 import BeautifulSoup
import pandas as pd
from pandas import *
from selenium import webdriver
import time
import re
from konlpy.tag import Kkma
from konlpy.tag import Okt
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import classification_report


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
kkma=Kkma()
okt=Okt()
from nltk.tokenize import word_tokenize
import nltk
from collections import Counter

driver= webdriver.Chrome("/users/solhee/data/chromedriver")
url="https://movie.naver.com/movie/bi/mi/pointWriteFormList.nhn?code=167491&type=after&isActualPointWriteExecute=false&isMileageSubscriptionAlready=false&isMileageSubscriptionReject=false"
url="https://movie.naver.com/movie/bi/mi/pointWriteFormList.nhn?code=185917&type=after&isActualPointWriteExecute=false&isMileageSubscriptionAlready=false&isMileageSubscriptionReject=false"
url="https://movie.naver.com/movie/bi/mi/pointWriteFormList.nhn?code=109960&type=after&isActualPointWriteExecute=false&isMileageSubscriptionAlready=false&isMileageSubscriptionReject=false"
driver.get(url)
movie=pd.DataFrame(columns=['score','text'])

error=[]
for i in range(1,11):
    print(1)
    driver.find_element_by_xpath('//*[@id="pagerTagAnchor{}"]'.format(i)).click()
    time.sleep(4)
    html=driver.page_source
    soup=BeautifulSoup(html,"html.parser")
    n=0
    for j in soup.select('div.score_result > ul > li'):
        try:
            print(n)
            lst = []
            lst.append(j.select_one('div.star_score').get_text(strip=True))
            lst.append(j.select_one('div:nth-of-type(2) > p > span#_filtered_ment_{}'.format(n)).get_text(strip=True))
            movie = movie.append(pd.Series(lst, index=movie.columns), ignore_index=True)
            n+=1
        except Exception as err:
            print(err)
            error.append([i,j])
            pass
movie.to_csv("/users/solhee/data/movie_mulan.csv", index=False)
movie.to_csv("/users/solhee/data/movie_bando.csv",index=False)
movie.to_csv("/users/solhee/data/movie_batman.csv",index=False)

driver.close()

driver.find_element_by_xpath('//*[@id="pagerTagAnchor{}"]'.format(110)).click()

# 담보/ 뮬란/ 반도/ 배트맨 2000개씩 합치기
dambo=read_csv("/users/solhee/data/movie.csv")
dambo=dambo.iloc[1:2000,:]

mulan=read_csv("/users/solhee/data/movie_mulan.csv")
bando=read_csv("/users/solhee/data/movie_bando.csv")
batman=read_csv("/users/solhee/data/movie_batman.csv")

dambo.append(mulan)
movie=pd.concat([dambo,mulan,bando,batman])

#1. 4 개 영화 리뷰 전처리
movie['text'].str.findall('[.]+')
def FIND(arg):
    return movie['text'].str.findall(arg)
FIND('[.]+')
movie['text'].str.replace('[.,]+','')
def mreplace(arg1,arg2):
    return movie['text'].str.replace(arg1,arg2)
movie['text']=mreplace('[.,!~ㅋㅎㅠ]','')
movie['text']=mreplace('오배건','오백원')
movie['text'][FIND('재밋다').apply(lambda x:x!=[])]
movie['text']=mreplace('재밋다','재밌다')
movie['text'].at[2026]='재밌긴한데 중반부가 좀 지루한 듯'

#영어는 대문자로 모두 통일
movie['text'][movie['text'].str.findall('dc',re.I).apply(lambda x:x!=[])]
movie['text']=movie['text'].str.upper()

movie['text'][FIND('[\^]+').apply(lambda x:x!=[])]
movie['text']=mreplace('[\^]','')
movie['text'][FIND('잇엇\w+').apply(lambda x:x!=[])]
movie=movie.reset_index(inplace=False)
movie['text'].at[7428]=movie['text'][7428].replace('멋잇엇고','멋있었고')
movie['text'][FIND('봣').apply(lambda x:x!=[])]
movie['text']=mreplace('봣어요','봤어요')
movie['text']=mreplace('자밋게','재밌게')
movie['text']=mreplace('봣는대','봤는데')
movie['text'].at[6945]='난 재밌게 봤는데 모르고 보는 사람은 이해가 안갈 듯'
movie['text'][178]
movie['text']=mreplace('봣지만','봤지만')
movie['text']=mreplace('봣','봤')

movie.to_csv("/users/solhee/data/movie_save_final.csv",index=False)
movie=read_csv("/users/solhee/data/movie_save_final.csv")
movie.info()
# 띄어쓰기 안된거 찾기
movie['text'][237]
#space :  평 빈칸
#space=movie['text'][pd.isna(FIND(' '))].index
movie['text'][pd.isna(movie['text'])]

# 평 안적혀져 있는 리뷰 삭제
movie=movie.drop(space,axis=0)
FIND('재밋\w+')
movie['text'][FIND('재밋\w+').apply(lambda x:x!=[])]
movie['text']=mreplace('재밋','재밌')
#save (갱신)
pd.set_option("max_rows",10000)
movie['text']
FIND('[ㅏ-ㅣ]+')
movie['text']=mreplace('[ㄱ-ㅎ]','')
movie['text']=mreplace('[ㅏ-ㅣ]','')
FIND('10자')
movie['text']=mreplace('10자','')
FIND('[?]+')
movie['text']
movie['text']=mreplace('[?]+','')
FIND('봫\w+')
movie['text'][FIND('봫\w+').apply(lambda x:x!=[])]
movie['text'].at[7857]='꿀잼이구요 재밌게 잘봤어요'
movie['text'][7857]
movie

#형태소 별
movie.to_csv("/users/solhee/data/movie_save_final2.csv",index=False)
movie=read_csv("/users/solhee/data/movie_save_final2.csv")
movie.info()
del movie['index']
len(movie[movie['score']>=7]['score'])/len(movie)

#7점이상을 긍정으로 6점이하 부정
movie['plma']=['긍정' if i>=7 else '부정' for i in movie['score']]
movie[movie['score']<=6]

movie[movie['score']]
X=movie['text']
Y=movie['plma']
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, stratify=Y)

X_train[pd.isna(X_train)]
del X_train[2337]
del Y_train[2337]

Y_test.value_counts()
Y_train.value_counts()


X_train[pd.isna(X_train)]
del X_train[2337]
del Y_train[2337]


#1. nltk 형태소
#test='나는 배가 부르다'
#okt.pos(test)
#['/'.join(i) for i in okt.pos(test)]

# 명사만 뽑기
#[i for i in okt.pos(test) if i[1]=='Noun']
#[j for j in okt.pos(i[0]) for i in movie['text']]
#[j for i in movie['text'] for j in okt.pos(i)]
#[okt.pos(i) for i in movie['text']]

def tokenize (doc):
    return ['/'.join(i) for i in okt.pos(doc)]
data=np.array([X_train,Y_train]).T.tolist()
data[1]

train_doc=[(tokenize(x[0]),x[1]) for x in data]
for i in train_doc:
    print(i)

# 단어장 생성
token=[j for i in train_doc for j in i[0]]

def term_exists(doc):
    return {word:(word in set(doc)) for word in token}

x_train=[(term_exists(d),c) for d,c in train_doc]
classifier=nltk.NaiveBayesClassifier.train(x_train)

classifier.show_most_informative_features()

X_test
tokenize(X_test[1])

test_doc=[tokenize(x) for x in X_test]
test_f=[{j:(j in token) for j in i} for i in test_doc]
pred=Series(classifier.classify(x) for x in test_f)
print(classification_report(Y_test,pred))

f=open("/users/solhee/data/nltk_pred_naive.csv","w")
f.write(classification_report(Y_test,pred))
f.close()

#2 . sklearn
movie=read_csv("/users/solhee/data/movie_save_final2.csv")
del movie['index']
movie.info()
movie['plma']=['긍정' if i>=7 else '부정' for i in movie['score']]
movie[pd.isna(movie['text'])]
movie=movie.drop(2337,axis=0)

def text_tokenizing(doc):
    return [word for word in okt.morphs(doc) if len(word)>1 and word not in stopword]
stopword=[]
with open("/users/solhee/data/stopword_kr.txt") as file:
    for word in file:
        stopword.append(word.strip())
contents_token=[text_tokenizing(i) for i in movie['text']]
contents=[" ".join(i) for i in contents_token]
X=contents
Y=movie['plma']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,stratify=Y)

# 부정/ 긍정 비율 확인
list(Counter(Y_train).values())[0]/list(Counter(Y_train).values())[1]
list(Counter(Y_test).values())[1]/list(Counter(Y_test).values())[0]

#택하기
cv=CountVectorizer()
x_train=cv.fit_transform(X_train)
nb=MultinomialNB()
nb.fit(x_train,Y_train)

x_test=cv.transform(X_test)
y_predict=nb.predict(x_test)
print(classification_report(Y_test,nb.predict(x_test)))

f=open("/users/solhee/data/vectorize_naive.csv","w")
f.write(classification_report(Y_test,nb.predict(x_test)))
f.close()

xy=(X_train,X_test,Y_train,Y_test)
np.save("/users/solhee/data/naive_bayes_splitdata.npy",xy)

X_train,X_test,Y_train,Y_test=np.load("/users/solhee/data/naive_bayes_splitdata.npy",allow_pickle=True)

# 오답 병합
Y_test_index=Y_test[y_predict!=Y_test].index
len(movie['text'][Y_test_index])
len(Y_test[y_predict!=Y_test])

X_train=X_train + movie['text'][Y_test_index].tolist()
Y_train=Y_train.tolist()+movie['plma'][Y_test_index].tolist()
Y_train=pd.Series(Y_train)
type(Y_train)
len(X_train)
len(Y_train)

## 틀린거 train 으로 합쳐버림 ( 단 test 데이터 에서 앞에꺼는 제외하지 않음 )
x_train=cv.fit_transform(X_train)
nb=MultinomialNB()
nb.fit(x_train,Y_train)

x_test=cv.transform(X_test)
y_predict=nb.predict(x_test)
print(classification_report(Y_test,nb.predict(x_test)))
f=open("/users/solhee/data/vectorize_navie2.csv","w")
f.write(classification_report(Y_test,nb.predict(x_test)))
f.close()
nb.score(x_test,Y_test)

'''
새로운 데이터 : 영화 소리도 없이 리뷰 100개 
'''
# 수집
driver= webdriver.Chrome("/users/solhee/data/chromedriver")
url="https://movie.naver.com/movie/bi/mi/pointWriteFormList.nhn?code=187893&type=after&onlyActualPointYn=N&onlySpoilerPointYn=N&order=newest"

driver.get(url)
movie=pd.DataFrame(columns=['score','text'])
error=[]
for i in range(1,11):
    print(1)
    driver.find_element_by_xpath('//*[@id="pagerTagAnchor{}"]'.format(i)).click()
    time.sleep(4)
    html=driver.page_source
    soup=BeautifulSoup(html,"html.parser")
    n=0
    for j in soup.select('div.score_result > ul > li'):
        try:
            print(n)
            lst = []
            lst.append(j.select_one('div.star_score').get_text(strip=True))
            lst.append(j.select_one('div:nth-of-type(2) > p > span#_filtered_ment_{}'.format(n)).get_text(strip=True))
            movie = movie.append(pd.Series(lst, index=movie.columns), ignore_index=True)
            n+=1
        except Exception as err:
            print(err)
            error.append([i,j])
            pass
movie.to_csv("/users/solhee/data/movie_test.csv", index=False)
driver.close()

movie_test=read_csv("/users/solhee/data/movie_test.csv")
#1. 정제 전 정확도
movie_test['plma']=['긍정' if i>=7 else '부정' for i in movie_test['score']]
movie_test_nan=movie_test[pd.isna(movie_test['text'])].index
movie_test=movie_test.drop(movie_test_nan,axis=0)
movie_test=movie_test.drop('score',axis=1)
contents_token_test=[text_tokenizing(i) for i in movie_test['text']]
contents_test=[" ".join(i) for i in contents_token_test]
X_movie_test=contents_test
Y_movie_test=movie_test['plma']

X_movie_test=cv.transform(X_movie_test)
Y_predict_test=nb.predict(X_movie_test)
print(classification_report(Y_movie_test,Y_predict_test))
print(nb.score(X_movie_test,movie_test['plma']))


###2. 정제 후 정확도 비교
####
def FIND(arg):
    return movie_test['text'].str.findall(arg)
def mreplace(arg1,arg2):
    return movie_test['text'].str.replace(arg1,arg2)
FIND('[.]+')
movie_test['text'].str.replace('[.,]+',' ')
movie_test['text']=mreplace('[.,!~ㅋㅎㅠ]',' ')
#movie_test['text'][FIND('재밋다').apply(lambda x:x!=[])]
movie_test['text']=movie_test['text'].str.upper()
movie_test['text'][FIND('[\^]+').apply(lambda x:x!=[])]
movie_test['text']=mreplace('[\^\?]',' ')
movie_test['text'][FIND('[ㄱ-ㅎ]+').apply(lambda x:x!=[])]
movie_test.at[86,'text']=' 30년만에 평점 처음써보는데진짜 진심 재미없네요' # 재미없ㄴㅔ요 -> 재미없네요
movie_test.at[22,'text']='이 정도면 쓰레기 영화가판치는 현 코로나상황에 비해 준수한 영화 유아인 연기 좋음' # 좋으ㅁ -> 좋음
movie_test.at[87,'text']='너무너무 재미없었어요 진짜' # .ㄹㅇ -> 진짜

movie_test['text']=mreplace('[ㄱ-ㅎ]','')
movie_test['text'][FIND('[ㅏ-ㅢ]+').apply(lambda x:x!=[])]
movie_test['text']=mreplace('[ㅏ-ㅢ]','')

contents_token_test=[text_tokenizing(i) for i in movie_test['text']]
contents_test=[" ".join(i) for i in contents_token_test]
X_movie_test=contents_test
Y_movie_test=movie_test['plma']


'''
오답 데이터 병합 반복시키기 (정확도 어떻게 변화하는지 관찰, overfitting 유의)
'''
iscore=[]
acc=[]
new_test_acc=[]
for i in range(1,21):
    Y_test_index = Y_test[y_predict!= Y_test].index #오답 인덱스
    X_train = X_train + movie['text'][Y_test_index].tolist()
    Y_train= Y_train.tolist() + movie['plma'][Y_test_index].tolist()
    Y_train = pd.Series(Y_train)
    x_train = cv.fit_transform(X_train)
    nb = MultinomialNB()
    nb.fit(x_train, Y_train) # 모델 생성
    x_test = cv.transform(X_test) # test 데이터 벡터로 변환
    y_predict = nb.predict(x_test) # 테스트 데이터 예측
    iscore.append(i)
    acc.append(nb.score(x_test, Y_test))
    x_movie_test = cv.transform(X_movie_test)
    Y_predict_test = nb.predict(x_movie_test)
    new_test_acc.append(nb.score(x_movie_test,movie_test['plma']))
len(X_train)
len(Y_train)

#정규화 시켜서 그래프 그리기
def min_max(arg):
    return (arg-arg.min())/(arg.max()-arg.min())
feature_acc=min_max(pd.Series(acc))
feature_new_test_acc=min_max(pd.Series(new_test_acc))

plt.plot(iscore,feature_acc,label="train")
plt.plot(iscore,feature_new_test_acc,label="new data")
plt.title("오답을 train 데이터에 반복 학습시킨 정확도")
plt.xlabel("반복 횟수")
plt.ylabel("accuracy")
plt.legend()

DATA=pd.DataFrame({"model_accuracy":acc,"new_data_accuracy":new_test_acc},index=iscore)
DATA.to_csv("/users/solhee/data/Naive_Bayes_naver_accuracy")