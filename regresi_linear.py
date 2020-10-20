import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title('Fuel Consumption CO2')
st.subheader('Simple Linear Regression by IBM & Fiona Stefani Limin')


#sidebar
st.sidebar.markdown("# Feature Selection")

feature_name = st.sidebar.selectbox(
    'Select Feature',
    ('Fuel Consumption', 'Engine Size', 'Cylinders')
)


#fetch some data
DATA_URL = ('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/FuelConsumptionCo2.csv')

@st.cache
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    return data

# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
data = load_data(10000)
# Notify the reader that the data was successfully loaded.
data_load_state.text("Done! ")

#inspect the raw data
st.subheader('Raw data')
st.write(data) #st.dataframe(data)

#data describe
st.subheader('Lets first have a descriptive exploration on our data')
if st.checkbox('Show Summarize'):
    #st.subheader('Raw data')
    st.write(data.describe())
if st.checkbox('Show Feature'):
    cdf = data[['enginesize','cylinders','fuelconsumption_comb','co2emissions']]
    st.write(cdf.head(9))



#histogram
#st.subheader('Plot each of the feature ')

#hist_values = np.histogram(
#    data['enginesize'], range=(0,24))[0]

#st.bar_chart(hist_values)


#plot 
st.write(f"### {feature_name} Feature Plot")
st.write(" Now, lets plot each of these features vs the Emission, to see how linear is their relation:")
fig = plt.figure()
cdf = data[['enginesize','cylinders','fuelconsumption_comb','co2emissions']]

def get_feature(name):
    x = None
    if name == 'Fuel Consumption':
        x = cdf.fuelconsumption_comb
    elif name == 'Engine Size':
        x = cdf.enginesize
    else:
        x = cdf.cylinders
    return x


plt.scatter(get_feature(feature_name), cdf.co2emissions,  color='blue')
plt.xlabel(f"{feature_name}")
plt.ylabel("Emission")
st.pyplot(fig)

#sidebar
import streamlit as st

# Add a selectbox to the sidebar:
add_selectbox = st.sidebar.selectbox(
    'Evaluation Metric',
    ('MAE', 'MSE', 'R2-Score')
)

# You can use a column just like st.sidebar:
st.write(f"### Engine Size Train data distribution")

left_column, right_column = st.beta_columns(2)

#creating train and test dataset
with left_column:
    C = st.slider('Slide to spilt data training and testing', 0.01, 0.8)
    msk = np.random.rand(len(data)) < C
    train = data[msk]
    test = data[~msk]

    fig = plt.figure()
    plt.scatter(train.enginesize, train.co2emissions,  color='blue')
    plt.xlabel("Engine size")
    plt.ylabel("Emission")
    st.pyplot(fig)

    from sklearn import linear_model
    regr = linear_model.LinearRegression()
    train_x = np.asanyarray(train[['enginesize']])
    train_y = np.asanyarray(train[['co2emissions']])
    regr.fit (train_x, train_y)



# Or even better, call Streamlit functions inside a "with" block:
with right_column:
    st.write(f"Coefficients: {regr.coef_}")
    st.write(f"Intercept: {regr.intercept_}")

    fig = plt.figure()
    plt.scatter(train.enginesize, train.co2emissions,  color='blue')
    plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
    plt.xlabel("Engine size")
    plt.ylabel("Emission")
    st.pyplot(fig)
 
# You can use a column just like st.sidebar:
st.write(f"### Evaluation")

from sklearn.metrics import r2_score

test_x = np.asanyarray(test[['enginesize']])
test_y = np.asanyarray(test[['co2emissions']])
test_y_ = regr.predict(test_x)

def get_evaluation(name):
    x = None
    if name == 'MAE':
        x = "Mean absolute error : %.2f" % np.mean(np.absolute(test_y_ - test_y))
    elif name == 'MSE':
        x = "Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2)
    else:
        x = "R2-score: %.2f" % r2_score(test_y , test_y_)
    return x

st.write(f" {get_evaluation(add_selectbox )}")

##author
st.write(f"### Author")
st.write("Fiona Stefani Limin")