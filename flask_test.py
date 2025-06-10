from flask import Flask

app = Flask(__name__)

@app.before_first_request
def load_once():
    print("✅ Loaded once before first request")

@app.route("/")
def home():
    return "Hello"

if __name__ == '__main__':
    app.run(debug=True)
