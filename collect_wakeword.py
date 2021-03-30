import datetime
import os
import time
import logging
from ctypes import *
from contextlib import contextmanager

import pyaudio
import wave


fs = 44100
seconds = 3
chunk = 1024
channels = 1
sample_format = pyaudio.paInt16

ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)

def py_error_handler(filename, line, function, err, fmt):
    pass

c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)

@contextmanager
def noalsaerr():
    asound = cdll.LoadLibrary('libasound.so')
    asound.snd_lib_error_set_handler(c_error_handler)
    yield
    asound.snd_lib_error_set_handler(None)

class WakeWordListener:

    def __init__(self):
        with noalsaerr():
            self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paInt16,
                                  channels=channels,
                                  rate=fs,
                                  frames_per_buffer=chunk,
                                  input=True)

    def close_stream(self):
        try:
            self.stream.stop_stream()
            self.stream.close()
            self.p.terminate()  
        except:
            pass

    def get_dir_name(self,):
        path = 'data/wake_word_test'
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def get_next_file_name(self):
        return os.path.join(self.get_dir_name(), str(datetime.datetime.now()) + '.wav')

    def save_data(self, frames):
        wf = wave.open(self.get_next_file_name(), 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(self.p.get_sample_size(sample_format))
        wf.setframerate(fs)
        wf.writeframes(b''.join(frames))
        wf.close()


def start_listening():

    try:
        while True:
            wake_word_listener = WakeWordListener()
            frames = []
            input(f'Press enter to record {seconds} seconds of wake word')
            time.sleep(0.1)
            print('Recording...')
            for i in range(int(fs/chunk*seconds)):
                readed = wake_word_listener.stream.read(
                    chunk, exception_on_overflow=False)
                frames.append(readed)
            wake_word_listener.close_stream()
            wake_word_listener.save_data(frames)

    except KeyboardInterrupt as e:
        print('closing recording')

    except Exception as e:
        raise e
        print(e)
    finally:
        print('closing')


start_listening()
