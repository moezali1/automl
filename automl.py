import streamlit as st
import pandas as pd
from PIL import Image

image = Image.open('queens_image.jpg')

def run():
    st.title("Smith AutoML")
    st.image(image)

    st.sidebar.info('This is AutoML app')
    selection = st.sidebar.selectbox("What would you like to do?", ['Model Training', 'Prediction'])

    if selection == 'Model Training':

        st.header("Upload CSV File for Model Training")
        file_upload = st.file_uploader("Upload CSV File", type=["csv"])
        
        if file_upload is not None:
            data = pd.read_csv(file_upload)
            st.dataframe(data)

            st.header("Statistical Analysis of Data")
            st.dataframe(data.describe())

            st.header("Select Your Target Variable")
            use_case = st.selectbox("use-case", ['Classification', 'Regression'])
            
            if use_case == 'Classification':
                from pycaret.classification import setup, pull, get_config, compare_models, plot_model, save_model, load_model, predict_model
            else:
                from pycaret.regression import setup, pull, get_config, compare_models, plot_model, save_model, load_model, predict_model
            
            target = st.selectbox("Target", data.columns)
            init_setup = st.button('Initialize Setup')

            if init_setup:
                s = setup(data, target = target, silent = True)
                setup_info = pull()
                st.dataframe(setup_info.data)

                st.header("Transformed Training Data")
                transformed_training = get_config('X_train')
                st.dataframe(transformed_training)

                st.header("Model Training")
                with st.spinner(text='Training Multiple Models. Please hold tight!'):
                    best_model = compare_models()
                    save_model(best_model, 'best_model')
                    results = pull()
                    st.dataframe(results)
                    st.balloons()
                    st.success("Best Model saved as best_model.pkl successfully!")

                st.header("Model Analysis")

                if use_case == 'Classification':
                    try:
                        plot_model(best_model, plot = 'auc', display_format='streamlit')
                    except:
                        st.write("AUC plot not available.")
                    plot_model(best_model, plot = 'confusion_matrix', display_format='streamlit')
                    plot_model(best_model, plot = 'class_report', display_format='streamlit')
                else:
                    plot_model(best_model, plot = 'residuals', display_format='streamlit')
                    plot_model(best_model, plot = 'error', display_format='streamlit')
                    plot_model(best_model, plot = 'cooks', display_format='streamlit')

    elif selection == 'Prediction':
        st.header("Upload CSV File for Prediction")
        file_upload_prediction = st.file_uploader("Upload CSV File", type=["csv"])
        if file_upload_prediction is not None:
            data = pd.read_csv(file_upload_prediction)
            st.dataframe(data)
            use_case = st.selectbox("What are you trying to do?", ['Classification', 'Regression'])

            if use_case == 'Classification':
                from pycaret.classification import load_model, predict_model
            else:
                from pycaret.regression import load_model, predict_model
            
            predict_button = st.button("Generate Predictions")

            if predict_button:
                loaded_model = load_model('best_model')
                predictions = predict_model(loaded_model, data=data)
                predictions.to_csv('predictions.csv')
                st.dataframe(predictions)
                st.balloons()
                st.success("predictions.csv successfully downloaded!")
            
if __name__ == '__main__':
    run()