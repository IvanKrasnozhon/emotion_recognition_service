from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from transformers import RobertaTokenizerFast, TFRobertaForSequenceClassification, pipeline

tokenizer = RobertaTokenizerFast.from_pretrained("arpanghoshal/EmoRoBERTa")
model = TFRobertaForSequenceClassification.from_pretrained("arpanghoshal/EmoRoBERTa")
emotion = pipeline('sentiment-analysis', model='arpanghoshal/EmoRoBERTa')

emotion_groups = {
    'Love': 'Happy',
    'Admiration': 'Happy',
    'Joy': 'Happy',
    'Approval': 'Happy',
    'Caring': 'Concerned',
    'Excitement': 'Happy',
    'Amusement': 'Happy',
    'Gratitude': 'Happy',
    'Desire': 'Happy',
    'Anger': 'Angry',
    'Optimism': 'Happy',
    'Disapproval': 'Angry',
    'Grief': 'Sadness',
    'Annoyance': 'Angry',
    'Pride': 'Happy',
    'Curiosity': 'Concerned',
    'Neutral': 'Neutral',
    'Disgust': 'Angry',
    'Disappointment': 'Angry',
    'Realization': 'Stare',
    'Fear': 'Concerned',
    'Relief': 'Happy',
    'Confusion': 'Concerned',
    'Remorse': 'Concerned',
    'Embarrassment': 'Concerned',
    'Surprise': 'Stare',
    'Sadness': 'Concerned',
    'Nervousness': 'Concerned'
}

class EmotionAnalysisHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        text = post_data.decode('utf-8')

        emotion_labels = emotion(text)

        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

        print(emotion_labels[0]['label'])

        response_data = {
            "emotion_label": emotion_groups[str.capitalize(emotion_labels[0]['label'])]
        }

        self.wfile.write(json.dumps(response_data).encode('utf-8'))

def run():
    server_address = ('', 8080)
    httpd = HTTPServer(server_address, EmotionAnalysisHandler)
    print('Starting the server on port 8080...')
    httpd.serve_forever()

if __name__ == '__main__':
    run()
