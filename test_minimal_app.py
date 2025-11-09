from flask import Flask

app = Flask(__name__)

@app.route("/")
def index():
    return "Medical Chatbot is running!"

@app.route("/test")
def test():
    return "Test endpoint working!"

if __name__ == '__main__':
    print("Starting minimal Flask app...")
    app.run(host="0.0.0.0", port=8080, debug=False)