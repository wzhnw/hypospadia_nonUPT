import streamlit as st  
import pandas as pd  
import pickle  
import matplotlib.pyplot as plt  
import numpy as np  
from sklearn.preprocessing import StandardScaler   
import shap

# Set page title and icon  
st.set_page_config(page_title="RF Prediction for postoperative complication of non-UPT", page_icon=":bar_chart:")  

# Add a title and subheader  
st.title("RF Prediction for postoperative complication of non-UPT")  
st.subheader("Enter patient information and click Submit to get the prediction")  

# Create a sidebar for input fields  
with st.sidebar:  
    st.header("Input Information")  
    Age = st.number_input("Age (months)")  
    length_of_glans_ = st.number_input("Length of glans (mm)")  
    width_of_glans = st.number_input("Width of glans (mm)")  
    UP_width = st.number_input("UP_width (mm)")  
    preoperative_curvature = st.number_input("Preoperative curvature (°)")  
    length_of_dificient_urethra = st.number_input("Length of deficient urethra (cm)")  
    
    submit_button = st.button("Submit")  

# If button is pressed  
if submit_button:  
    try:  
        # Load the saved model from the file  
        with open(r'C:\Users\77434\Desktop\wzh\RF_UPT\RF.pkl', 'rb') as f:
            clf = pickle.load(f)  

        # Store inputs into dataframe  
        X = pd.DataFrame([[Age, length_of_glans_, width_of_glans,  UP_width,
                           preoperative_curvature, length_of_dificient_urethra]],   
                         columns=['Age', 'length_of_glans_', 'width_of_glans',  'UP_width',
                                  'preoperative_curvature', 'length_of_dificient_urethra'])  

        # 选择需要进行标准化的列  
        columns_to_scale = ['Age','length_of_glans_', 'width_of_glans', 'UP_width',
                    'preoperative_curvature', 'length_of_dificient_urethra']  

        # 实例化 StandardScaler  
                # Load the complete dataset for SHAP analysis  
        BT_complete = pd.read_csv(r'C:\Users\77434\Desktop\wzh\RF_UPT\BF_complete.csv')
        BT= pd.read_csv(r'C:\Users\77434\Desktop\wzh\RF_UPT\UPT_BF.csv')
        c = BT_complete[['Age', 'length_of_glans_', 'width_of_glans',  'UP_width',
                         'preoperative_curvature', 'length_of_dificient_urethra']]  
        b=BT[['Age', 'length_of_glans_', 'width_of_glans',  'UP_width',
                         'preoperative_curvature', 'length_of_dificient_urethra'  
                         ]] 
        
        scaler = StandardScaler()


        # 拟合 StandardScaler 并转换整个数据集
        scaled_df = scaler.fit(b)

        # 假设您有一个单个样本，需要进行标准化
        # 这里我们使用 df 的最后一行作为示例单个样本
        single_sample = X[columns_to_scale] # 将单个样本转换为适合 transform 的格式

        # 使用之前拟合的 StandardScaler 对单个样本进行标准化
        scaled_single_sample = scaler.transform(single_sample)

        # 将标准化后的单个样本转换为 DataFrame 格式
        X1 = pd.DataFrame(scaled_single_sample, columns= columns_to_scale)
        
        
        # Generate predictions  
        prediction = clf.predict(X1)  

        # Initialize SHAP explainer  
        dt_explainer_df = shap.KernelExplainer(clf.predict, c)  
        
        # Calculate SHAP values  
        shap_values_df = dt_explainer_df.shap_values(X1.values)


        # Plot SHAP summary and display it in Streamlit  
        st.subheader("Prediction Result")
        st.caption("'1' means occurrence of postsurgery complications,'0' means no occurrence of posturgery complications")
        st.write(f"Predicted Result: {prediction[0]}")  # Display prediction  

        st.divider()
    
        # Create a waterfall plot for the first prediction  
        st.subheader("SHAP Value Plot")


        #Create SHAP waterfall plot
        expl = shap.Explanation(values=shap_values_df[0], base_values=dt_explainer_df.expected_value,
                                data=X1.iloc[0])
        shap.waterfall_plot(expl, max_display=10)
        plt.title("SHAP Waterfall Plot")  
        
        # Display the plot in Streamlit  
        st.pyplot(plt)
        plt.clf()

    except Exception as e:  
        st.error(f"Error occurred: {e}")