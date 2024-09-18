from flask import Flask, request, jsonify
import requests

app = Flask(__name__)


def fetch_conversion_rate(source_currency, amount, target_currency):
    URL = "https://free.currconv.com" + "/api/v7/convert?q={},{}&compact=ultra&callback=sampleCallback&apiKey={}".format(
        source_currency, target_currency, "e0b5b5b2b5b5b2b5b5b2")

    response = requests.get(URL)
    data = response.json()
    print(data)

    return data["{}_{}".format(source_currency, target_currency)]


@app.route('/', methods=['POST'])
def index():
    data = request.get_json()
    source_currency = data['queryResult']['parameters']['unit-currency']['currency']
    amount = data['queryResult']['parameters']['unit-currency']['amount']
    target_currency = data['queryResult']['parameters']['currency-name']
    print(source_currency, amount, target_currency)

    cf = fetch_conversion_rate(source_currency, amount, target_currency)
    final_amount = round(amount * cf, 2)

    response = {
        "fulfillmentText": "{} {} is {} {}".format(amount, source_currency, final_amount, target_currency)
    }

    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True)
