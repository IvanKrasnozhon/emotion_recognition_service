from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from transformers import RobertaTokenizerFast, TFRobertaForSequenceClassification, pipeline

tokenizer = RobertaTokenizerFast.from_pretrained("arpanghoshal/EmoRoBERTa")
model = TFRobertaForSequenceClassification.from_pretrained("arpanghoshal/EmoRoBERTa")
emotion = pipeline('sentiment-analysis', model='arpanghoshal/EmoRoBERTa')

class EmotionAnalysisHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        text = post_data.decode('utf-8')

        emotion_labels = emotion(text)

        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

        response_data = {
            "emotion_labels": emotion_labels
        }

        self.wfile.write(json.dumps(response_data).encode('utf-8'))

def run():
    server_address = ('', 8080)
    httpd = HTTPServer(server_address, EmotionAnalysisHandler)
    print('Starting the server on port 8080...')
    httpd.serve_forever()

if __name__ == '__main__':
    run()
