import streamlit as st
import pandas as pd
import numpy as np
from sklearn import svm
import pickle
from sklearn.feature_extraction.text import CountVectorizer

st.set_page_config(
    page_title= 'Hotel Review Classification',
    page_icon='😄'
)
st.title('Hotel Review Classifier')
st.image('hotel.jpg')
sentence = str(st.text_input('Feedback : '))
def process(sentence):
    d=pd.read_csv('tripadvisor_hotel_reviews.csv')
    d_neg=d.loc[d['Rating']<=3]
    d_neg.reset_index(drop=True)
    d_five=d.loc[d['Rating']>=4]
    d_five.reset_index(drop=True)
    d_all=pd.concat([d_neg,d_five],axis=0)
    d_all=d_all.reset_index(drop=True)
    d_all['Sentiment']=np.where(d_all['Rating']>=4,'Positive','Negative')

    from sklearn.model_selection import train_test_split
    v = CountVectorizer()
    x_train,x_test,y_train,y_test = train_test_split(d_all.Review,d_all.Sentiment,test_size = 0.2,random_state=2)
    x_train_vec = v.fit_transform(x_train)
    x_test_vec = v.transform(x_test)
    clf_svm = svm.SVC(kernel = 'linear')
    clf_svm.fit(x_train_vec,y_train)

    rev = [sentence]
    rev_vec = v.transform(rev)
    op = clf_svm.predict(rev_vec)
    return op[0]

if st.button("submit"):
    if (process(sentence))=='Positive':
        st.markdown(process(sentence))
        st.balloons()
    else:
        st.markdown(process(sentence))

st.caption('By Pradnya More')


