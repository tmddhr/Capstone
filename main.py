import streamlit as st
import pandas as pd
import customers as cs


def main():
    st.title('Shop Customer Data Prediction')

    df = pd.read_csv('Customers-2.csv')
    st.dataframe(cs.displayHeadRows(df))

    st.title('size of train and test subsets after splitting')
    st.dataframe(cs.test_train(df))

    st.title('Classifiers comparison')
    st.dataframe(cs.classifiers(df))

    st.title('prediction of future customers')
    st.dataframe(cs.predictions(df).head(10))


if __name__ == '__main__':
    main()