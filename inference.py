import pyaudio
import threading
import time
import torch
import numpy as np
from pydub import AudioSegment
from model import SimpleRNN, SimpleCNN
from datasets import get_extractor

class MicListener:

    def __init__(self, record_length=2, sample_rate=8000):

        p = pyaudio.PyAudio()
        self.chunk = 1024
        self.stream = p.open(format=pyaudio.paInt16,
                            channels=1,
                            rate=sample_rate,
                            input=True,
                            output=True,
                            frames_per_buffer=self.chunk)
    
    def listen(self, queue):
        while True:
            data = self.stream.read(self.chunk, exception_on_overflow=True)
            queue.append(np.frombuffer(data, dtype=np.int16))
            time.sleep(0.1)

    
    def run(self, queue):
        thread = threading.Thread(target=self.listen, args=(queue, ), daemon=False)
        thread.start()

        print('Listener is on...\n')


class WakeWord:
    def __init__(self, model_path: str):
        self.mic_listener = MicListener()
        self.model = torch.load(model_path)
        self.model.eval().to('cpu')
        self.extractor = get_extractor()
        self.current_audio_queue = []
    
    def _read_audio(self, audio_path):
        silent_audio = AudioSegment.silent(duration=4000)
        audio = silent_audio.overlay(AudioSegment.from_wav(audio_path))
        audio = np.frombuffer(audio.set_frame_rate(8000).set_channels(1)[0:4000]._data, dtype='int16')
        audio = torch.Tensor(audio.reshape(1, -1))
        return audio

    def predict(self, audio):
        with torch.no_grad():
            audio = np.concatenate(audio)
            audio = torch.Tensor(audio).reshape(1, -1)
            data = self.extractor(audio)
            print(data.shape) 
            output = self.model(data)
            predictions = torch.round(torch.sigmoid(output))
            return predictions.item()

    
    def inference(self):
        while True:
            if len(self.current_audio_queue) > 32:
                diff = len(self.current_audio_queue) - 32
                for _ in range(diff):
                    self.current_audio_queue.pop(0)
                    print('predicting...')
                    predicted = self.predict(self.current_audio_queue)
                    
                    print('predicted ', predicted)
                    if int(predicted) == 1:
                        print('Hello', '='*10)
                    # print(self.predict(self.current_audio_queue))
            elif len(self.current_audio_queue) == 32:
                print('predicting...')
                print(self.current_audio_queue)
                print(np.concatenate(self.current_audio_queue))
                predicted = self.predict(self.current_audio_queue)
                
                    
                print('predicted ', predicted)
                if int(predicted) == 1:
                    print('Hello')
            time.sleep(0.1)
    
    def run(self,):
        self.mic_listener.run(self.current_audio_queue)
        thread = threading.Thread(target=self.inference, daemon=False)
        thread.start()


if __name__ == '__main__':
    wake_word = WakeWord('wake_model_cnn.pth')

    wake_word.run()