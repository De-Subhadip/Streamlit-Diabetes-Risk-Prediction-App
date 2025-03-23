import joblib
import xgboost
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(layout='wide')

st.header(':rainbow[Welcome to Diabetes Risk Prediction App]:wave:', divider=True)



filePath = 'diabetes_dataset.csv'

@st.cache_data
def load_data():
    data = pd.read_csv(filePath)
    return data

st.subheader(":blue-background[Raw data:]")
df = load_data()

dietMapping = {
    "0" : 'Unbalanced',
    "1" : 'Balanced',
    "2" : 'Veg'
}

df['DietType'] = df['DietType'].astype(str)
df['DietType'] = df['DietType'].map(dietMapping)

dfInfo = pd.DataFrame(
    {
        'Column' : [col for col in df.columns],
        'Null Count' : [df[col].isnull().sum() for col in df.columns],
        'Data Type' : [str(df[col].dtype) for col in df.columns]
    }
)

if st.checkbox('**Show Column Description:**'):
    colDescrip1, colDescrip2, colDescrip3 = st.columns(3)

    with colDescrip1:
        with st.container(border=True):
            st.markdown("""
                        :green[1. Age: ]The age of the individual (18-90 years).\n
                        :green[2. Pregnancies: ]Number of times the patient has been pregnant.\n
                        :green[3. BMI (Body Mass Index): ]A measure of body fat based on height and weight (kg/m²).\n
                        :green[4. Glucose: ]Blood glucose concentration (mg/dL), a key diabetes indicator.\n
                        :green[5. BloodPressure: ]Systolic blood pressure (mmHg), higher levels may indicate hypertension.\n
                        :green[6. HbA1c: ]Hemoglobin A1c level (%), representing average blood sugar over months.
            """)
    
    with colDescrip2:
        with st.container(border=True):
            st.markdown("""
                        :green[7. LDL (Low-Density Lipoprotein): ]'Bad' cholesterol level (mg/dL).\n
                        :green[8. HDL (High-Density Lipoprotein): ]'Good' cholesterol level (mg/dL).\n
                        :green[9. Triglycerides: ]Fat levels in the blood (mg/dL), high values increase diabetes risk.\n
                        :green[10. WaistCircumference: ]Waist measurement (cm), an indicator of central obesity.\n
                        :green[11. HipCircumference: ]Hip measurement (cm), used to calculate WHR.
            """)

    with colDescrip3:
        with st.container(border=True):
            st.markdown("""
                        :green[12. WHR (Waist-to-Hip Ratio): ]Waist circumference divided by hip circumference.\n
                        :green[13. FamilyHistory: ]Indicates if the individual has a family history of diabetes (1 = Yes, 0 = No).\n
                        :green[14. DietType: ]Dietary habits (Unbalanced, Balanced, Veg).\n
                        :green[15. Hypertension: ]Presence of high blood pressure (1 = Yes, 0 = No).\n
                        :green[16. MedicationUse: ]Indicates if the individual is taking medication (1 = Yes, 0 = No).\n
                        :green[17. Outcome: ]Diabetes diagnosis result (1 = Diabetes, 0 = No Diabetes).
            """)
        

st.dataframe(df, height=600)

if st.toggle("**Show Dataframe Info:**"):
    colInfo1, colInfo2, colInfo3 = st.columns(3)
    with colInfo2:
        st.dataframe(dfInfo, hide_index=True, height=628)

st.markdown("[Link to Kaggle Dataset](https://www.kaggle.com/datasets/asinow/diabetes-dataset)")
st.divider()



numCols = [col for col in df.columns if df[col].nunique() > 10]
nominalCols = [col for col in df.columns if col not in numCols]



col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.subheader(':blue-background[Descriptive Statistics:]')

    @st.cache_data
    def descriptiveStatsData():
        data = df[numCols].describe()
        return data

    dfDescriptiveStats = descriptiveStatsData()
    st.dataframe(dfDescriptiveStats)
st.divider()



st.subheader(':blue-background[Some Data Visualizations:]')
colCountPlot, colHistogram, colScatterPlot = st.columns(3, border=True)

with colCountPlot:
    st.subheader(':blue[Count Plot:]')

    columnForCountPlot = st.selectbox(
        label = 'Choose the column for which you want to view the countplot:',
        options = nominalCols,
        index = 1
    )

    fig1 = px.histogram(df, x=df[columnForCountPlot].astype(str), text_auto=True, color=columnForCountPlot, title=f"Count plot of {columnForCountPlot}")
    countPlot = st.plotly_chart(fig1)

with colHistogram:
    st.subheader(':red[Histogram:]')

    columnForHistogram = st.selectbox(
        label = 'Choose the column for which you want to view the histogram:',
        options = numCols,
        index = 3
    )

    fig2 = px.histogram(df, x=columnForHistogram, title=f"Count plot of {columnForHistogram}", color_discrete_sequence=['indianred'])
    histogram = st.plotly_chart(fig2)

with colScatterPlot:
    st.subheader(':orange[Scatter Plot:]')

    x = st.selectbox(
        label = 'Choose the column for X-axis',
        options = numCols,
        index = 4
    )

    y = st.selectbox(
        label = 'Choose the column for Y-axis',
        options = numCols,
        index = 3
    )

    fig3 = px.scatter(df, x=x, y=y, color_discrete_sequence=['orange'])
    scatterPlot = st.plotly_chart(fig3)

st.divider()



st.subheader(':blue-background[Preprocessed Dataframe:]')

st.markdown("""
            **The following preprocessing steps were used before fitting the XGBoost model on the data:**\n
            :green[1. ] The raw data has been scaled using a :blue[_Sklearn MinMaxScaler()_] object.\n
            :green[2. ] The categorical column :blue[_DietType_] has been one-hot enoded.\n
            :green[3. ] The :blue[_WaistCircumference_] and :blue[_HipCircumference_] columns has been dropped because the column :blue[_WHR_]\
            is a ratio of those two.\n
            :green[4. ] The features and labels were also separated before fitting the model on the data.
""")

@st.cache_resource
def load_scaler():
    return joblib.load('scaler.pkl')

scaler = load_scaler()

X = df.drop(['Outcome', 'WaistCircumference', 'HipCircumference'], axis=1)
y = df['Outcome']

colsToScale = [col for col in X.columns if X[col].nunique() > 10]

X[colsToScale] = scaler.transform(X[colsToScale])
X = pd.get_dummies(X, columns=["DietType"], dtype='int8')

if st.toggle('**Show preprocessed dataframe:**'):
    colPreprocessed1, colPreprocessed2 = st.columns([4, 1])

    with colPreprocessed1:
        st.write(":blue-background[Features : X]")
        st.dataframe(X)
    with colPreprocessed2:
        st.write(":blue-background[Labels : Y]")
        st.dataframe(y)

st.divider()



st.subheader(':blue-background[Making Predictions:]')
colInput, colPredButton, colPrediction = st.columns([2, 1, 2], vertical_alignment='center')

predDict = {}

with colInput:

    with st.container(height=400):
        st.markdown(':red-background[Please input your values for prediction:]')
        
        age = st.number_input(label=':orange[Enter your age:]', min_value=18, max_value=90, step=1, placeholder='Enter the age...', key='age')
        predDict["Age"] = st.session_state.age

        pregnancies = st.slider(label=':orange[Choose number of pregnancies (if any):]', min_value=0, max_value=16, step=1, key='pregnancies')
        predDict["Pregnancies"] = st.session_state.pregnancies

        bmi = st.number_input(label=':orange[Enter your BMI:]', min_value=15.0, max_value=50.0, step=0.01, key='bmi')
        predDict['BMI'] = st.session_state.bmi

        glucose = st.number_input(label=':orange[Enter your glucose level:]', min_value=50.0, max_value=210.0, step=0.1, key='glucose')
        predDict['Glucose'] = st.session_state.glucose

        bloodPressure = st.slider(label=':orange[Select your blood pressure:]', min_value=60, max_value=150, key='bloodPressure')
        predDict['BloodPressure'] = st.session_state.bloodPressure

        hba1c = st.slider(label=':orange[Select your HbA1c]', min_value=4.0, max_value=7.0, step=0.1, key='hba1c')
        predDict['HbA1c'] = st.session_state.hba1c

        ldl = st.number_input(label=':orange[Enter your LDL:]', min_value=-12.0, max_value=205.0, step=0.1, key='ldl')
        predDict['LDL'] = st.session_state.ldl

        hdl = st.number_input(label=':orange[Enter your HDL:]', min_value=-9.0, max_value=110.0, step=0.1, key='hdl')
        predDict['HDL'] = st.session_state.hdl

        triglycerides = st.number_input(label=':orange[Enter tryglecerides level:]', min_value=50.0, max_value=350.0, step=0.01, key='triglycerides')
        predDict['Triglycerides'] = st.session_state.triglycerides

        waist = st.number_input(label=':orange[Enter waist circumference]', min_value=40.0, max_value=350.0, step=0.1, key='waist')
        predDict['WaistCircumference'] = st.session_state.waist

        hip = st.number_input(label=':orange[Enter hip circumference]', min_value=50.0, max_value=160.0, step=0.1, key='hip')
        predDict['HipCircumference'] = st.session_state.hip

        familyHistory = st.pills(label=':orange[Select 1 if there is a family history of diabetes:]', options=[0, 1], default=0, key='familyHistory')
        predDict['FamilyHistory'] = st.session_state.familyHistory

        diet = st.pills(label=':orange[Choose the type of diet:]', options=['Balanced', 'Unbalanced', 'Veg'], default='Balanced', key='diet')
        predDict['DietType'] = st.session_state.diet

        hypertension = st.pills(label=':orange[Select 1 if you have hypertension:]', options=[0, 1], default=0, key='hypertension')
        predDict['Hypertension'] = st.session_state.hypertension

        medication = st.pills(label=':orange[Select 1 if you take medications:]', options=[0, 1], default=0, key='medication')
        predDict['MedicationUse'] = st.session_state.medication

dfPred = pd.DataFrame([predDict.values()], columns=list(predDict.keys()))
dfPred = pd.get_dummies(dfPred, columns=['DietType'])
dfPred['WHR'] = dfPred['WaistCircumference']/dfPred['HipCircumference']
dfPred = dfPred.drop(['WaistCircumference', 'HipCircumference'], axis=1)

for col in X.columns:
    if col not in dfPred.columns:
        dfPred[col] = 0

dfPred = dfPred[[col for col in X.columns]]

dfPred[colsToScale] = scaler.transform(dfPred[colsToScale])

@st.cache_resource
def load_model():
    return joblib.load('model.pkl')

model = load_model()

container = colPrediction.container(height=100, border=True)

def makePrediction(model, dfPred):
    pred = model.predict(dfPred)

    predProb = model.predict_proba(dfPred)
    n = np.argmax(predProb, axis=1)
    chance = predProb[0][n]

    if pred == 1:
        container.subheader(':red[High risk of diabetes]:warning:')
        container.write(f":red[Probability of suffering from diabetes:] :red{predProb[0][n]}")
    else:
        container.subheader(':green[Low risk of diabetes]:large_green_circle:')
        container.write(f":green[Probability of not suffering from diabetes:] :green{predProb[0][n]}")


with colPredButton:
    predButton = st.button(label='Click to Predict Diabetes', on_click=makePrediction, args=(model, dfPred, ))