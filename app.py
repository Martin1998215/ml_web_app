import streamlit as st
import pandas as pd
import numpy as np
from streamlit_option_menu import option_menu
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets

#dataset

# dataset for diabetes
diabetes = pd.read_csv("C:/Users/Dell/Desktop/mlapp/diabetes.csv")


X_diabetes = diabetes.drop("Outcome",axis=1)
Y_diabetes = diabetes["Outcome"]
X_diabetes_train,X_diabetes_test,Y_diabetes_train,Y_diabetes_test = train_test_split(X_diabetes,Y_diabetes,random_state=2,test_size=0.2)

#train model for Diabetes prediction
clf_diabetes = RandomForestClassifier(n_estimators=300)
clf_diabetes.fit(X_diabetes_train,Y_diabetes_train)

# Heart dataset
heart = pd.read_csv("C:/Users/Dell/Desktop/mlapp/heart.csv")
X_heart = heart.drop("target",axis=1)
Y_heart = heart["target"]

X_heart_train,X_heart_test,Y_heart_train,Y_heart_test = train_test_split(X_heart,Y_heart)

#train model for Heart Disease prediction
clf_heart = RandomForestClassifier(n_estimators=300)
clf_heart.fit(X_heart_train,Y_heart_train)



#setup sidebar
with st.sidebar:
    selected = option_menu("Menu List",
    ["Home","Diabetes Analysis","Diabetes Model","Heart Disease Analysis","Heart Model"],
    icons=["house-door","file-bar-graph","activity","file-bar-graph","heart"],
    default_index=0)

st.sidebar.write('''
**AI For Medicine**

Predicitive Modeling

Automating Medical Diagnosis

**Martin Sichibeya**.

''')    

#conditional statements

#For Home section
if selected == "Home":
    st.header("Machine Learning App")
    st.subheader("AI For Medicine ML App")
    st.write('''
    
    **Here we will make two predictions:**
    1. On Diabetes Dataset
    2. On Heart Disease Dataset 

    ''')
    
    @st.cache
    def convert_df(df):
        
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv().encode('utf-8')


    csv = convert_df(diabetes)
    csv1 = convert_df(heart)
    st.write("**Download Diabetes Data**")
    st.download_button(
     label="Download Diabetes CSV",
     data=csv,
     file_name='Diabetes.csv',
     mime='text/csv',
 )

    st.write("**Download Heart Disease Data**")
    st.download_button(
     label="Download Heart disease CSV",
     data=csv1,
     file_name='heart.csv',
     mime='text/csv',
 )

# Add image
    from PIL import Image
    img = Image.open("mlcover.jpg")
    st.write("**Profile Photo**")
    st.image(img,caption="ML Cover Photo")

    with open("mlcover.jpg", "rb") as file:
        
        btn = st.download_button(
             label="Download image",
             data=file,
             file_name="cover.jpg",
             mime="image/jpg"
           )
    


if selected == "Diabetes Analysis":
    st.header("Diabetes Data Analysis")
    st.subheader("DataFrame")
    st.write(diabetes)
    st.subheader("Columns")
    st.write(diabetes.columns)
    st.text("Number of Rows in data")
    st.write(diabetes.shape[0])
    st.subheader("Check for info")
    st.write(diabetes.info())
    st.subheader("Check for missing values")
    st.write(diabetes.isnull().sum())
    st.subheader("Split data into Features and Target values")

    st.subheader("Feature Values")
    st.write(X_diabetes)
    st.subheader("Target Values")
    st.write(Y_diabetes)
    st.text("Target Value count")
    st.write(Y_diabetes.value_counts())

    st.subheader("Split data into training and testing data")

    st.write("Original Data", X_diabetes.shape)
    st.write("Training Data", X_diabetes_train.shape)
    st.write("Test Data", X_diabetes_test.shape)

    # Accuracy score
    diabetes_train_score = clf_diabetes.score(X_diabetes_train,Y_diabetes_train)
    st.write("Training Score", diabetes_train_score * 100,'%')

    diabetes_test_score = clf_diabetes.score(X_diabetes_test,Y_diabetes_test)
    st.write("Test Score", diabetes_test_score * 100,'%')


if selected == "Diabetes Model":
    st.header("Diabetes Model Prediction") 
    st.subheader("Input Features...")

    #input features
    pregnancies = st.text_input("Pregnancies")
    glucose = st.text_input("Glucose")
    bp = st.text_input("Blood Pressure")
    skin_t = st.text_input("Skin Thickness")
    insulin = st.text_input("Insulin")
    bmi = st.text_input("BMI")
    pdf = st.text_input("Diabetes Pedigree Function")
    age = st.text_input("Age")



    # Enter button
    if st.button("Enter For Prediction"):
        # make predictions on input data
        diabetes_predict = clf_diabetes.predict([[pregnancies,glucose,bp,skin_t,insulin,bmi,pdf,age]])

        # prediction probability
        diabetes_predict_proba = clf_diabetes.predict_proba([[pregnancies,glucose,bp,skin_t,insulin,bmi,pdf,age]])
        if diabetes_predict[0] == 1:
            st.subheader("Results")
            st.success("You have Diabetes..")
            st.subheader("Prediction Probability")
            st.write("0 = Not Diabetic; 1 = Diabetic")
            st.write(diabetes_predict_proba)
        else:
            st.subheader("Results")
            st.success("You dont have Diabetes..")
            st.subheader("Prediction Probability")
            st.write("0 = Not Diabetic [%]; 1 = Diabetic [%]")
            st.write(diabetes_predict_proba * 100)    



if selected == "Heart Disease Analysis":
    st.header("Heart Disease Data Analysis")
    st.subheader("DataFrame")
    st.write(heart)
    st.subheader("Columns")
    st.write(heart.columns)
    st.text("Number of Rows in data")
    st.write(heart.shape[0])
    st.subheader("Check for info")
    st.write(heart.info())
    st.subheader("Check for missing values")
    st.write(heart.isnull().sum())
    st.subheader("Split data into Features and Target values")

    st.subheader("Feature Values")
    st.write(X_heart)
    st.subheader("Target Values")
    st.write(Y_heart)
    st.text("Target Value count")
    st.write(Y_heart.value_counts())

    st.subheader("Split data into training and testing data")

    st.write("Original Data", X_heart.shape)
    st.write("Training Data", X_heart_train.shape)
    st.write("Test Data", X_heart_test.shape)

    # Accuracy score
    heart_train_score = clf_heart.score(X_heart_train,Y_heart_train)
    st.write("Training Score", heart_train_score * 100,'%')

    heart_test_score = clf_heart.score(X_heart_test,Y_heart_test)
    st.write("Test Score", heart_test_score * 100,'%')


if selected == "Heart Model":
    st.header("Iris Model Prediction.") 
    st.subheader("Input Features...")

    age = st.text_input("Age")
    sex = st.text_input("Sex: Male:1, Female:0")
    cpt = st.text_input("Chest Pain Type (4 values)")
    rbp = st.text_input("Resting Blood Pressure")
    sc = st.text_input("Serum Cholestoral in mg/dl")
    fbs = st.text_input("Fasting Blood Sugar > 120 mg/dl")
    rer = st.text_input("Rest Electrocardiographic Results")
    mhra = st.text_input("Maximum Heart Rate Achieved")
    eia = st.text_input("Exercise Induced Angina")
    oldpeak = st.text_input("Oldpeak")
    slope = st.text_input("Slope of the peak Exercise")
    numb = st.text_input("Number of Blood Vessels colered by flourosopy")
    thal = st.text_input("Thal: 3 = Normal; 6 = Fixed Defect; 7 = Reversable Defect")


    if st.button("Enter Button"):


        heart_predict = clf_heart.predict([[age,sex,cpt,rbp,sc,fbs,rer,mhra,eia,oldpeak,slope,numb,thal]])

    
        # heart_predict_proba = clf_1.predict_proba([[age,sex,cpt,rbp,sc,fbs,rer,mhra,eia,oldpeak,slope,numb,thal]])
        heart_predict_proba = clf_heart.predict_proba([[age,sex,cpt,rbp,sc,fbs,rer,mhra,eia,oldpeak,slope,numb,thal]])


        if heart_predict[0] == 1:
            st.subheader("Results")
            st.success("You have heart disease")
            st.subheader("Predicition Probability")
            st.write("0 = No Heart Disease[%], 1 = Heart Disease [%]")
            st.write(heart_predict_proba * 100)
        else:
            st.subheader("Results")
            st.success("You DONT have heart disease")
            st.subheader("Prediction Probability")
            st.write("0 = No Heart Disease[%], 1 = Heart Disease [%]")
            st.write(heart_predict_proba * 100)

