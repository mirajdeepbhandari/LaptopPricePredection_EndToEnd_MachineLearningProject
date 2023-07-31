import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import streamlit as st


df=pd.read_csv("laptopdataset.csv")

# Splitting the data into features (X) and target variable (y)
y = df.pop('Price')
X = df

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=2)

# Scaling the selected features using MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train[['Ram', 'Weight', 'ppi', 'clock speed (Ghz)', 'SSD']])


# Replacing the scaled features in the training and testing sets
X_train[['Ram', 'Weight', 'ppi', 'clock speed (Ghz)', 'SSD']] = X_train_scaled


# Creating an instance of Linear Regression model
lr = LinearRegression()

# Fitting the model on the training data
lr.fit(X_train, y_train)

# Streamlit web app
st.title("Laptop Price Prediction")
st.markdown("""
    <h2 style="color: white; margin-top: 30px; ">About:</h2>
""", unsafe_allow_html=True)


st.markdown("""
    <div class="explanation-container">
        This web application enables users to estimate the price of laptops in the , <b style="color:#9c8148">Indian market</b>. By entering specific laptop specifications, including brand, type, GPU, processor, operating system, RAM, weight, PPI, and SSD capacity, the application predicts the price in <b style="color:#9c8148">Indian Rupees</b>. It offers valuable insights into potential laptop costs based on the provided configurations, aiding users in making informed decisions while considering various options available in the Indian market.
    </div>
    """, unsafe_allow_html=True)


st.markdown("""
        <style>
                    
        .explanation-container{
                    
           background-color:#0b0a12; 
           padding:20px;
           text-align:justify;
            border-radius:20px;
            color:#bababa;
             font-family: 'Roboto', cursive;
            font-size:17px;
            line-height:35px;
            margin-top: 30px;
            margin-bottom: 30px;

        }
        .main{
             background-color:#181b1b;
        }

        .css-fg4pbf{
            color:#41f6ce;
        }  

        #laptop-price-prediction,#enter-laptop-specifications{
            color:white; 
        }   
            
        .css-ue6h4q{
            color:#f7db30;
        }
            
        .st-br,.st-bs{
            color:white;
        }
        code{
           background-color:black;
            color:white;
        }
        .st-bw{
            background-color:#070a10;
        }
        .st-ej div{
            color:black;
        }
        .st-bx {
            background-color:black;
            }
        [data-testid="stTickBarMin"],[data-testid="stTickBarMax"]{
            color:white;
        }
      .css-7ym5gk{
            color:black;
      }
        .st-ed {
           color: #ffdc7d;
        }
            

        #MainMenu{
            visibility:hidden;
            }
    footer{
            
         visibility:hidden;   
    }
    header{
         visibility:hidden;    
    }
         </style>
        """, unsafe_allow_html=True)








# Input form for user to enter laptop specifications
st.header("Enter Laptop Specifications:")


company_options = ['Choose Company','Acer', 'Apple', 'Asus', 'Chuwi', 'Dell', 'Fujitsu', 'Google', 'HP', 'Huawei', 'Lenovo', 'LG', 'Mediacom', 'Microsoft', 'MSI', 'Razer', 'Samsung', 'Toshiba', 'Vero', 'Xiaomi']
company = st.selectbox("Company", company_options)
st.write("Selected company:", company)

useful_company = ['Apple', 'Chuwi', 'Dell', 'Google', 'HP', 'Mediacom', 'Microsoft', 'Toshiba', 'Vero']
                  

if company in useful_company:
    company_dict = {
        'Company_Apple': [0],
        'Company_Chuwi': [0],
        'Company_Dell': [0],
        'Company_Google': [0],
        'Company_HP': [0],
        'Company_Mediacom': [0],
        'Company_Microsoft': [0],
        'Company_Toshiba': [0],
        'Company_Vero': [0],
        
    }
    company_key = "Company_" + company
    company_dict[company_key] = [1]
else:
    company_dict = {
        'Company_Apple': [0],
        'Company_Chuwi': [0],
        'Company_Dell': [0],
        'Company_Google': [0],
        'Company_HP': [0],
        'Company_Mediacom': [0],
        'Company_Microsoft': [0],
        'Company_Toshiba': [0],
        'Company_Vero': [0],
    }
 


#for options
type_options = ['Notebook', 'Workstation','Gaming','None']
type = st.radio("Select Laptop Type:", type_options,index=3)

type_dict={
   'TypeName_Notebook':[0],
   'TypeName_Workstation':[0]
 }

if type == 'Notebook':
    type_dict['TypeName_Notebook'] = [1]
    type_dict['TypeName_Workstation'] = [0]


elif type == 'Workstation':
     type_dict['TypeName_Notebook'] = [0]
     type_dict['TypeName_Workstation'] = [1]
    
st.write("Selected Laptop Type:", type)



#for processor options
processor_options = [
    'Choose Processor',
    'Intel Core i3',
    'Intel Core i5',
    'Intel Core i7',
    'Other Intel processors',
    'AMD processor',
    'Apple Processor',
]

processor = st.selectbox("Processor", processor_options)
st.write("Selected Processor:", processor)

useful_processor = ['Intel Core i3', 'Intel Core i5', 'Intel Core i7', 'Other Intel processors']

if processor in useful_processor:
    processor_dict = {
        'processor_Intel Core i3': [0],
        'processor_Intel Core i5': [0],
        'processor_Intel Core i7': [0],
        'processor_Other Intel processors':[0]
    }

    processor_key = "processor_" + processor
    processor_dict[processor_key] = [1]
else:
    processor_dict = {
        'processor_Intel Core i3': [0],
        'processor_Intel Core i5': [0],
        'processor_Intel Core i7': [0],
        'processor_Other Intel processors':[0]
          
    }
 


#for processor options
os_options = ['macOS','Mac OS X','Windows 7','Windows 10', 'Linux', 'Android','Chrome OS','Windows 11','Others','None']

os = st.selectbox("Operating System", os_options,index=9)
st.write("Selected Operating System:", os)

useful_os = ['Windows 10', 'Windows 7']

if os in useful_os:
    os_dict = {
        'OpSys_Windows 10': [0],
        'OpSys_Windows 7': [0],
    }

    os_key = "OpSys_" + os
    os_dict[os_key] = [1]
else:
    os_dict = {
      'OpSys_Windows 10': [0],
      'OpSys_Windows 7': [0],
          
    }


ram_options = [2, 4, 6, 8, 12, 14]
ram = st.selectbox("RAM (GB)", ram_options, index=0)  # Set index 3 (8GB) as the default option
#To display the selected ram
st.write("Selected RAM:", ram, "GB")


weight = st.slider("Weight (kg)", 0.5, 4.0, 1.5)
st.write("Selected Weight", weight, "kg")

ppi = st.slider("Pixels Per Inch (PPI)", 90.5, 205.0, 100.0)
st.write("Selected PPI", ppi)

clock_speed = st.slider("Clock Speed (GHz)", 0.95, 4.0, 3.3)
st.write("Clock Speed", clock_speed,"GHz")


ssd_options = [0, 8, 16, 32, 64, 128, 180, 240, 256, 512, 640]
ssd = st.selectbox("SSD Capacity (GB)", ssd_options, index=7)  
st.write("SSD Capacity:", ssd, "GB")


# Create a Predict button
if st.button("Predict"):
    # Check if any required inputs are not selected
    if company == "Choose Company" or processor == "Choose Processor" or os == "None":
        st.warning("Please select all required laptop specifications before predicting.")
    else:
        # Convert input data to DataFrame
        df_company = pd.DataFrame(company_dict)
        df_type = pd.DataFrame(type_dict)
        df_processor = pd.DataFrame(processor_dict)
        df_os = pd.DataFrame(os_dict)
        
        non_catogorical_columns = {
            "Ram": [ram],
            "Weight": [weight],
            "ppi": [ppi],
            "clock speed (Ghz)": [clock_speed],
            "SSD": [ssd]
        }
        df_numericalcolumns = pd.DataFrame(non_catogorical_columns)
        scaled_val = scaler.transform(df_numericalcolumns)
        df_numericalcolumns_scaled = pd.DataFrame(scaled_val, columns=df_numericalcolumns.columns)
        
        # Concatenate the dataframes
        testdf = pd.concat([df_numericalcolumns_scaled, df_company, df_type, df_processor, df_os], axis=1)

        # Perform prediction using the trained model
        y_pred = lr.predict(testdf)

        y_pred_original = np.expm1(y_pred)

        # Display the predicted price to the user
        st.write("Predicted Price:", "Rs.", int(y_pred_original[0]))
        
st.markdown("""
    <h5 style="color:white; margin-top: 30px; font-family: 'Roboto', cursive; text-decoration:underline;">Developed by Miraj Deep Bhandari</h5>
""", unsafe_allow_html=True)

      