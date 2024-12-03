from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from scipy.stats import norm
import asyncio

app = FastAPI()

# Mount the static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates directory
templates = Jinja2Templates(directory="templates")


def build_model(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    x = LSTM(100, return_sequences=True)(inputs)
    x = Dropout(0.2)(x)
    x = LSTM(100, return_sequences=False)(x)
    x = Dropout(0.2)(x)
    x = Dense(50, activation='relu')(x)
    outputs = Dense(1)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


async def async_predict(stock_code, prediction_days):
    try:
        # Fetch data asynchronously
        loop = asyncio.get_running_loop()
        data = await loop.run_in_executor(None, lambda: yf.download(stock_code, start='2023-01-01', end='2024-11-01'))

        if data.empty:
            return {"error": f"No data available for stock code: {stock_code}"}

        data = data['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        # Prepare training data
        sequence_length = 200
        x_train, y_train = [], []

        for i in range(sequence_length, len(scaled_data) - prediction_days):
            x_train.append(scaled_data[i - sequence_length:i, 0])
            y_train.append(scaled_data[i:i + prediction_days, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        # Build and train the model asynchronously
        model = build_model((x_train.shape[1], x_train.shape[2]))
        await loop.run_in_executor(None, lambda: model.fit(x_train, y_train, epochs=10, batch_size=1, verbose=0))

        # Prepare test data (last sequence_length days for prediction)
        test_data = scaled_data[-sequence_length:]
        x_test = np.array([test_data])
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        # Predict future prices asynchronously
        predicted_price_scaled = await loop.run_in_executor(None, model.predict, x_test)
        predicted_prices = scaler.inverse_transform(predicted_price_scaled)

        # Ensure realistic price range by adding a minimum price difference
        low_price = np.percentile(predicted_prices, 10)
        high_price = np.percentile(predicted_prices, 90)
        if high_price <= low_price * 1.1:  # Add minimum margin to ensure meaningful difference
            high_price = low_price * 1.1

        # Calculate standard deviation and mean of predictions
        std_dev = np.std(predicted_prices)
        if std_dev == 0:
            std_dev = 0.01  # Avoid zero std deviation

        mean_price = np.mean(predicted_prices)

        # Calculate probabilities aiming for ~80% confidence interval
        low_prob = norm.cdf(low_price, loc=mean_price, scale=std_dev)
        high_prob = 1 - norm.cdf(high_price, loc=mean_price, scale=std_dev)

        # Ensure probabilities are within a realistic range and scale to percentages
        low_prob = min(max(low_prob * 100, 0), 100)
        high_prob = min(max(high_prob * 100, 0), 100)

        # Prepare response
        prediction_result = {
            'stock_code': stock_code,
            'low_price': float(low_price),
            'low_prob': float(low_prob),
            'high_price': float(high_price),
            'high_prob': float(high_prob),
        }

        return prediction_result
    except Exception as e:
        return {"error": str(e)}


@app.get('/', response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post('/predict')
async def predict(request: Request):
    form_data = await request.form()
    stock_code = form_data['stock_code']

    try:
        prediction_days = int(form_data['prediction_days'])
    except ValueError:
        return JSONResponse(content={"error": "Prediction days must be an integer."}, status_code=400)

    if prediction_days < 1 or prediction_days > 365:
        return JSONResponse(content={"error": "Prediction days must be between 1 and 365."}, status_code=400)

    prediction_result = await async_predict(stock_code, prediction_days)

    if "error" in prediction_result:
        return JSONResponse(content=prediction_result, status_code=400)

    return JSONResponse(content=prediction_result)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
