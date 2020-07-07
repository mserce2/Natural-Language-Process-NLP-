# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 17:34:05 2020

@author: Mete

Bu bölümde nlp ile atılan bir tweet'in bir kadın tarafından mı yoksa erkek tarafından mı olduğunu
tespit edip sınıflamalar yapmaya çalışacağız
"""
#%%
import nltk
nltk.download("punkt")

#%%libraries
import pandas as pd
data=pd.read_csv(r"gender_classifier.csv",encoding="latin1") #latin harfleri içerdiği için latin1 seçtik

#data içindeki verilerden sadece cinsiyet ve atılan tweet içeriğini alarak sınıflandırma yapacağız
data=pd.concat([data.gender,data.description],axis=1)
#gereksiz olan nan ifadelerini siliyoruz daha fazla verim almak adına
data.dropna(axis=0,inplace=True) #axis=0 ise satır axis=1 ise sütün  inplace=True ise yeni değişken tanımlamadan oto kayıt yap
#sınıflandırma yapmak için int verilerine çevirme işlemi yapıyoruz
#1 ise kadın 0 ise erkek
data.gender=[1 if each =="female" else 0 for each in data.gender]

#%%clening data    
#Datamız içinde yazı dışında karakterler olabilir(+-*/:);?&+% gibi) datamızı eğitirken bu gibi
#karakterleri yok etmemiz gerek ki algoritmamız max seviyede çalışabilsin.

import re
first_description =data.description[4] #data daki 4 .satır(spyder ide kullanmanızı tavsiye ederim değişiklikleri kolay fark edebilirsiniz)
description=re.sub("[^a-zA-Z]"," ",first_description) #(a dan z ye küçük harfler) A-Z (büyük harfler) bulma onun dışındaki karakterleri " "(space) ile değiştir
description=description.lower() #büyük harfleri küçük harflere çevirme

#%%stopwords(irrelavent words) gereksiz kelimelerden kurtulma
#örndeğin: i go to the school. burda ki the anlamı yok sadece grammer için kullanılır

import nltk #natural language tool kit
nltk.download("stopwords") #carpus diye bir klasöre indiriliyor
from nltk.corpus import stopwords #sonra carpus klasöründen import ediyoruz

#description=description.split()
#split yerine tokenizer kullanabiliriz daha fazla avantaj sağlar çünkü;
#"shouldn't ve guzel ".split yaparsak should not olarak algılamaz ancak nltk.tokenize algılar
description=nltk.word_tokenize(description)

#%%
#gereksiz kelimerden kurtul
description=[word for word in description if not word in set(stopwords.words("english"))]


#%%lemmatazation loved=>love gitmeyeceğim=>git yani kök bulma işlemi
#erkek:maça gitmek çok güzel,maç çok iyiydi,maçı kazandık burdaki tweette maç kelimesine gelen
#ekler bizim için anlam ifade etmiyor çünkü algoritmamız maç kelimesi ile eğitiliyor
#bu gibi fazlalıklardan kurtulmamız gerek
import nltk
nltk.download("wordnet")
import nltk as nlp

lemma=nlp.WordNetLemmatizer()
description=[ lemma.lemmatize(word) for word in description]

#kelimeleri birleştirip kontrol edelim
description=" ".join(description)



#%%az önceki işlemleri sadece bir satır için yapmıştık şimdi ise tü data için yapacağız

description_list=[]
for description in data.description:
    description=re.sub("[^a-zA-z]"," ",description)
    description=description.lower()
    description=nltk.word_tokenize(description)
    #description=[word for word in description if not word in set(stopwords.words("english"))] #uzun sürdüğü için devre dışı bıraktık ama aşağıda daha hızlı şekli vardır :)
    lemma=nlp.WordNetLemmatizer()
    description=[ lemma.lemmatize(word) for word in description]
    description=" ".join(description)
    description_list.append(description)




#%%bag of words 

from sklearn.feature_extraction.text import CountVectorizer #bag of words yaratmak için kullandığımız metod

max_features=7500 #bag of words resmindeki sütündaki kelimeler gibi anlayabiliriz en çok kullanılan 500 kelime seçiyoruz
count_vectorizer=CountVectorizer(max_features=max_features,stop_words="english") #yukarıda uzun zaman alığı için kapatmıştık burda kısa sürede biteceği için açtık

space_matrix=count_vectorizer.fit_transform(description_list).toarray() #bag of words resmindeki işlemi yaptık
print("en cok kullanılan:{} kelimeler:{}".format(max_features,count_vectorizer.get_feature_names()))


#%%
y=data.iloc[:,0].values  #male or female classes
x=space_matrix

#train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=42)

#%%naive bayes
from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
nb.fit(x_train,y_train)

#%%prediction
y_pred=nb.predict(x_test)

print("accuracy:",nb.score(y_pred.reshape(-1,1),y_test))














