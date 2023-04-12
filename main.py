import pickle
import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression , ElasticNet , Lasso , Ridge

df = pd.read_csv("bodyfat.csv")


X = df.drop(['BodyFat','Density'],axis=1)
y = df['Density']


X['Bmi']=703*X['Weight']/(X['Height']*X['Height'])
# X.head()

X['ACratio'] = X['Abdomen']/X['Chest']
X['HTratio'] = X['Hip']/X['Thigh']
X.drop(['Weight','Height','Abdomen','Chest','Hip','Thigh'],axis=1,inplace=True)
# X.head()

z = np.abs(stats.zscore(X))


# only keep rows with z-scores less than 3 
X_clean = X[(z<3).all(axis=1)]
y_clean = y[(z<3).all(axis=1)]



from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_clean,y_clean,random_state=42)



trans = PowerTransformer()
X_train = trans.fit_transform(X_train)
X_test = trans.transform(X_test)



linear = LinearRegression()
linear.fit(X_train,y_train)

def pred(age, weight, height, neck, chest, abdomen, hip, thigh, knee, ankle, biceps, forearm, wrist):
    bmi = weight/(height*height)
    ACratio = abdomen/chest
    HTratio = hip/thigh
    input_data = np.array([[age, neck,	knee,	ankle,	biceps,	forearm,	wrist,	bmi,	ACratio,	HTratio]])
    density = linear.predict(input_data)
    return density



# with open('model.pkl', 'rb') as f:
#     model = pickle.load(f)

# def pred(age, weight, height, neck, chest, abdomen, hip, thigh, knee, ankle, biceps, forearm, wrist):
#     bmi = weight/(height*height)
#     ACratio = abdomen/chest
#     HTratio = hip/thigh
#     input_data = np.array([[age, neck,	knee,	ankle,	biceps,	forearm,	wrist,	bmi,	ACratio,	HTratio]])
#     prediction = model.predict(input_data)
#     return prediction



# Density,BodyFat,Age,Weight,Height,Neck,Chest,Abdomen,Hip,Thigh,Knee,Ankle,Biceps,Forearm,Wrist
# 1.0708,12.3,23,154.25,67.75,36.2,93.1,85.2,94.5,59.0,37.3,21.9,32.0,27.4,17.1
# 74,207.50,70.00,40.8,112.4,108.5,107.1,59.3,42.2,24.6,33.7,30.0,20.9
def main():
    st.title("Body Fat %")

    age = st.number_input("Age", min_value=1, max_value=150, value=23)
    weight = st.number_input("Weight (kg)", min_value=10.00, max_value=500.000, value=154.25)
    height = st.number_input("Height (cm)", min_value=50.00, max_value=300.00, value=67.75)
    neck = st.number_input("Neck circumference (cm)", min_value=10.00, max_value=50.00, value=36.2)
    chest = st.number_input("Chest circumference (cm)", min_value=10.00, max_value=300.00, value=93.1)
    abdomen = st.number_input("Abdomen circumference (cm)", min_value=10.00, max_value=300.00, value=85.5)
    hip = st.number_input("Hip circumference (cm)", min_value=10.00, max_value=300.00, value=94.5)
    thigh = st.number_input("Thigh circumference (cm)", min_value=10.00, max_value=300.00, value=59.0)
    knee = st.number_input("Knee circumference (cm)", min_value=10.00, max_value=300.00, value=37.3)
    ankle = st.number_input("Ankle circumference (cm)", min_value=10.00, max_value=300.00, value=21.9)
    biceps = st.number_input("Biceps circumference (cm)", min_value=10.00, max_value=300.00, value=32.0)
    forearm = st.number_input("Forearm circumference (cm)", min_value=10.00, max_value=300.00, value=27.4)
    wrist = st.number_input("Wrist circumference (cm)", min_value=10.00, max_value=300.00, value=17.1)

    if st.button("Predict"):
        density = pred(age, weight, height, neck, chest, abdomen, hip, thigh, knee, ankle, biceps, forearm, wrist).reshape(1,-1)
        # st.success(f"Predicted age, weight, height and body measurements: {result:.1f}")
        fat = ((4.95/density[0]) - 4.5)*100
        st.text(fat[0])
        # st.text(density)
 
if __name__ == '__main__':
        main()
