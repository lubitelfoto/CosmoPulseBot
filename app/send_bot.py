import requests


def send_picture_to_telegram(bot_token, chat_id, photo_path):
    url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
    with open(photo_path, 'rb') as photo:
        files = {'photo': photo}
        data = {'chat_id': chat_id}
        response = requests.post(url, files=files, data=data)
    return response.json()


def send_text_to_telegram(bot_token, chat_id, text, parse_mode=None):
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    data = {'chat_id': chat_id, 'text': text, 'parse_mode': parse_mode}
    response = requests.post(url, data=data)
    return response.json()
