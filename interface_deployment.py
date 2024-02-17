import gradio as gr

outputs = gr.Textbox(label="Prediction")
def predict_close_and_class(open_val, low_val, high_val, volume_val):
    user_input = [[open_val, low_val, high_val, volume_val]]
    user_input_df = pd.DataFrame(user_input, columns=['open', 'low', 'high', 'volume'])

    # Scale user input features
    user_input_scaled = scaler.transform(user_input_df[['open', 'high', 'low', 'volume']])

    # Predict using Decision Tree Regressor model
    predicted_close_dt_regressor = dt_regressor.predict(user_input_scaled)[0]

    # Predict using Decision Tree Classifier model
    predicted_class_dt = dt_classifier.predict(user_input_df[['open', 'high', 'low', 'volume']])[0]

    # Return predictions as a dictionary
    return {'Predicted Close (Regressor)': predicted_close_dt_regressor, 'Predicted Class (Classifier)': predicted_class_dt}


# Create Gradio interface
iface = gr.Interface(
    fn=predict_close_and_class,
    inputs=['number', 'number', 'number', 'number'],
    outputs=outputs,
    title='Stock Close Price Prediction',
    description='Enter values for "open", "low", "high", and "volume" to predict the close price and class.'

)

# Launch the interface
iface.launch()

explain about gradio and also this code