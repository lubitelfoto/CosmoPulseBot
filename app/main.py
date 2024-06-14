import asyncio
import os

import requests
import json
from app.model_manager import ModelManager
import matplotlib.pyplot as plt

import pandas as pd
from send_bot import send_text_to_telegram, send_picture_to_telegram
import numpy as np

from scipy import fftpack



mod_manager = ModelManager()


def load_data():
    url = 'https://services.swpc.noaa.gov/products/solar-wind/plasma-7-day.json'
    response = requests.get(url)
    if response.status_code == 200:
        data = json.loads(response.text)
        print("Data loaded successfully")
    else:
        print("Data loading error")
        data = []
    df = pd.DataFrame(data[1:], columns=data[0])
    df[['density','speed','temperature']] = df[['density','speed','temperature']].astype(np.float32)
    df['time_tag'] = pd.to_datetime(df['time_tag'])
    max_date = df['time_tag'].max()
    min_date = max_date-pd.Timedelta(3, 'D')
    all_dates = pd.date_range(start=min_date, end=max_date, freq='T') # 'T' - minutely frequency
    all_dates_df = pd.DataFrame(all_dates, columns=['time_tag'])
    merged_df = pd.merge(all_dates_df, df, on='time_tag', how='left')
    three_days_ago = df['time_tag'].max() - pd.Timedelta(days=3)
    merged_df = merged_df[merged_df['time_tag'] > three_days_ago]
    merged_df = merged_df.infer_objects()
    merged_df.interpolate(method='linear', inplace=True)
    merged_df = merged_df.bfill()
    new_columns_list = []
    fourier_df = pd.DataFrame([merged_df['density'].values])
    for i in range(len(fourier_df)):
        sig = fourier_df.iloc[i].values
        win = np.kaiser(len(sig), 5)
        x_win = sig*win
        X_win = fftpack.fft(x_win)

        new_columns = np.hstack((sig, np.abs(X_win)))
        new_columns_list.append(new_columns)

    new_columns_df = pd.DataFrame(
        new_columns_list,
        columns=[f'new_col_{j+1}' for j in range(len(new_columns_list[0]))])
    fourier_df = pd.concat([fourier_df, new_columns_df], axis=1)
    X = fourier_df.to_numpy()
    return X, merged_df['time_tag']

async def predict_event():
    picture_path = "images/forecast.png"

    try:
        model = mod_manager.get_model()
        X, time_X = load_data()
        proba = model.predict_proba(X)
        prediction = model.predict(X)
        event_predict = "Storm possible" if prediction == 1 else "Storm unlikely"

        plt.figure(figsize=(10, 6))
        plt.plot(time_X, X[0][:4320], linestyle='-', color='green', alpha=0.5)
        plt.gca().set_facecolor('black')
        plt.gca().spines['bottom'].set_color('white')
        plt.gca().spines['top'].set_color('white')
        plt.gca().spines['left'].set_color('white')
        plt.gca().spines['right'].set_color('white')
        plt.gca().tick_params(axis='x', colors='white')
        plt.gca().tick_params(axis='y', colors='white')
        plt.gca().yaxis.label.set_color('white')
        plt.gca().xaxis.label.set_color('white')
        plt.title(f'{event_predict}, Probability of storm - {int(proba[0][1] * 100)}%', color='white')
        plt.xlabel('Time', color='white')
        plt.ylabel('Density', color='white')

        # Check if the 'images' directory exists, and create it if not
        if not os.path.exists("images"):
            os.mkdir("images")

        # Save the plot as an image
        plt.savefig(picture_path, facecolor='black')

        return f"Probability of storm - {proba[0][1]}", picture_path

    except Exception as e:
        return f"An error occurred: {e}"


async def send_s(chat_id):
    pass

async def main():
    print("App started")
    token = os.getenv('TELEGRAM')
    chat_id = '@cosmopulse_free'


    while True:
        message, picture_path = await predict_event()
        print(message)
        send_picture_to_telegram(bot_token=token, chat_id=chat_id, photo_path=picture_path)
        await asyncio.sleep(60 * 10)


if __name__ == '__main__':
    asyncio.run(main())



