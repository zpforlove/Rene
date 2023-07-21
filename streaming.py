import time
import pyaudio
from threading import Thread
from cfg_parse import cfg
import torch
import numpy as np
from model import Model
import torch.nn.functional as F
import whisper

FORMAT = pyaudio.paInt16
N_CHANNEL = 1
SR = 8000
CHUCK_SIZE = 1024
chuck_pos = 0
ring_length = 100 * 60 * 60
ring_buffer = [np.array([0] * CHUCK_SIZE)] * ring_length

class_ids = {}
with open('./class-id.txt', 'r') as fr:
    for line in fr.readlines():
        line = line.strip()
        lab, cls = line.split('|')
        class_ids[int(cls)] = lab


model = Model(cfg).to('cpu')
checkpoint = torch.load('./Rene.pth', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()


def whisper_feature(wav, model):
    '''
    Use OpenAI pre-trained whisper model
    :param wav: audio wav
    :param model: model type
    :return: whisper decoded audio feature
    '''
    model = whisper.load_model(model)
    audio = whisper.pad_or_trim(wav)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)
    feature = result.audio_features
    return feature


def recording(record_param):
    """ Real time audio recording
    :param record_param: Recording parameters
    :return: None
    """
    global ring_buffer
    global ring_length
    global chuck_pos

    p = pyaudio.PyAudio()
    stream = p.open(format=record_param['FORMAT'],
                    channels=record_param['CHANNELS'],
                    rate=record_param['SR'],
                    input=True,
                    frames_per_buffer=record_param['CHUCK_SIZE'])
    print('recording...')
    while 1:
        byte_stream = stream.read(record_param['CHUCK_SIZE'])
        data = np.frombuffer(byte_stream, np.int16).flatten().astype(np.float32) / 32768.0
        ring_buffer[chuck_pos] = data
        chuck_pos += 1
        if chuck_pos == ring_length:
            chuck_pos = 0
    stream.stop_stream()
    stream.close()
    p.terminate()


def feature_calc(audio):
    data = torch.FloatTensor(audio)
    feature = whisper_feature(data, 'tiny').to(torch.float32).cpu()
    return feature


def inference(feature):
    with torch.no_grad():
        input_lengths = torch.LongTensor([cfg['whisper_seq']] * 1)
        # pred = alpha * model(feature, input_lengths) + lgb_pred * (1 - alpha)
        pred = model(feature, input_lengths)
        log_prob = F.softmax(pred, dim=1).data.cpu().numpy()
        return log_prob


def predict(audio_from_mic):
    feature = feature_calc(audio_from_mic)
    pred_lab = inference(feature.unsqueeze(0))
    return pred_lab


def streaming(callback, n_sec=10):
    global ring_buffer
    global chuck_pos
    time.sleep(11)  # sleep 11 seconds
    print('streaming decoding running...')

    while 1:
        # Take 10s of audio from the ring_buffer
        pos = chuck_pos
        if pos - 100 * n_sec < 0:
            start_pos = 100 * n_sec - pos
            pre = np.concatenate(ring_buffer[-start_pos:])
            flo = np.concatenate(ring_buffer[:pos])
            data = np.concatenate((pre, flo))
        else:
            data = np.concatenate(ring_buffer[pos - 100 * n_sec:pos])

        # Calculate the results of disease classification
        ret = callback(data)
        prob = np.squeeze(ret)
        max_index = np.argsort(-prob).tolist()
        flush_log = ''
        for i, prob_idx in enumerate(max_index):
            if not i:
                flush_log += '\033[1;35m%s:%.2f%%\033[0m' % (class_ids[prob_idx], prob[prob_idx] * 100) + '\n'
            else:
                flush_log += '%s:%.2f%%' % (class_ids[prob_idx], prob[prob_idx] * 100) + '\n'
        print(flush_log)

        time.sleep(2)


if __name__ == '__main__':
    recording_args = {
        'ring_buffer': ring_buffer,
        'ring_length': ring_length,
        'FORMAT': FORMAT,
        'CHANNELS': N_CHANNEL,
        'SR': SR,
        'CHUCK_SIZE': CHUCK_SIZE
    }
    t_rec = Thread(target=recording, args=(recording_args,))
    t_predict = Thread(target=streaming, args=(predict, 10))
    t_rec.start()
    t_predict.start()
