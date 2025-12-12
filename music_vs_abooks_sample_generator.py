import os
import math
import random

from pydub import AudioSegment

FRAME_RATE = 8000
SAMPLE_LENGTH = 24000

count_per_type_train = 2000
count_per_type_validate = 500

files_per_type = 25

samples_per_file_train = count_per_type_train // files_per_type
samples_per_file_validate = count_per_type_validate // files_per_type

rnd = random.Random()

music_filenames = os.listdir('music_train')
print(len(music_filenames), music_filenames)

music_raw_data_train = bytes(0)
music_raw_data_validate = bytes(0)

for n in range(files_per_type):
    fn_in = f'music_train/{music_filenames[n]}'
    sound = AudioSegment.from_mp3(fn_in)

    print(n, fn_in, sound.frame_rate, sound.sample_width, sound.channels)
    print(len(sound.raw_data), len(sound.raw_data) / (sound.frame_rate * sound.sample_width * sound.channels))
    sound = sound.set_frame_rate(FRAME_RATE)
    sound = sound.set_channels(1)
    sound = sound.set_sample_width(1)
    print(len(sound.raw_data), len(sound.raw_data) / FRAME_RATE)

    total_len = len(sound.raw_data)

    i = 0
    while i < samples_per_file_train:
        start = 0.9 * rnd.random() + 0.05
        start = round(start * total_len)
        if total_len - start < 3 * SAMPLE_LENGTH:
            continue
        chunk_raw_data = sound.raw_data[start:start+SAMPLE_LENGTH]
        music_raw_data_train += chunk_raw_data
        i += 1

    i = 0
    while i < samples_per_file_validate:
        start = 0.9 * rnd.random() + 0.05
        start = round(start * total_len)
        if total_len - start < 3 * SAMPLE_LENGTH:
            continue
        chunk_raw_data = sound.raw_data[start:start + SAMPLE_LENGTH]
        music_raw_data_validate += chunk_raw_data
        i += 1
    
music_raw_sound_train = AudioSegment(music_raw_data_train, frame_rate=FRAME_RATE, sample_width=1, channels=1)

# music_raw_sound_train.export('music_train.pcm', format='u8')
music_raw_sound_train.export('music_train.mp3', format='mp3')

with open('music_train.raw', 'wb') as f:
    f.write(music_raw_data_train)

music_raw_sound_validate = AudioSegment(music_raw_data_validate, frame_rate=FRAME_RATE, sample_width=1, channels=1)

# music_raw_sound_validate.export('music_validate.pcm', format='u8')
music_raw_sound_validate.export('music_validate.mp3', format='mp3')

with open('music_validate.raw', 'wb') as f:
    f.write(music_raw_data_validate)

abooks_filenames = os.listdir('abooks_train')
print(len(abooks_filenames), abooks_filenames)

abooks_raw_data_train = bytes(0)
abooks_raw_data_validate = bytes(0)

for n in range(files_per_type):
    fn_in = f'abooks_train/{abooks_filenames[n]}'
    sound = AudioSegment.from_mp3(fn_in)

    print(n, fn_in, sound.frame_rate, sound.sample_width, sound.channels)
    print(len(sound.raw_data), len(sound.raw_data) / (sound.frame_rate * sound.sample_width * sound.channels))
    sound = sound.set_frame_rate(FRAME_RATE)
    sound = sound.set_channels(1)
    sound = sound.set_sample_width(1)
    print(len(sound.raw_data), len(sound.raw_data) / FRAME_RATE)

    total_len = len(sound.raw_data)

    i = 0
    while i < samples_per_file_train:
        start = 0.9 * rnd.random() + 0.05
        start = round(start * total_len)
        if total_len - start < 3 * SAMPLE_LENGTH:
            continue
        chunk_raw_data = sound.raw_data[start:start + SAMPLE_LENGTH]
        abooks_raw_data_train += chunk_raw_data
        i += 1

    i = 0
    while i < samples_per_file_validate:
        start = 0.9 * rnd.random() + 0.05
        start = round(start * total_len)
        if total_len - start < 3 * SAMPLE_LENGTH:
            continue
        chunk_raw_data = sound.raw_data[start:start + SAMPLE_LENGTH]
        abooks_raw_data_validate += chunk_raw_data
        i += 1

abooks_raw_sound_train = AudioSegment(abooks_raw_data_train, frame_rate=FRAME_RATE, sample_width=1, channels=1)

# abooks_raw_sound_train.export('abooks_train.pcm', format='u8')
abooks_raw_sound_train.export('abooks_train.mp3', format='mp3')

with open('abooks_train.raw', 'wb') as f:
    f.write(abooks_raw_data_train)

abooks_raw_sound_validate = AudioSegment(abooks_raw_data_validate, frame_rate=FRAME_RATE, sample_width=1, channels=1)

# abooks_raw_sound_validate.export('abooks_validate.pcm', format='u8')
abooks_raw_sound_validate.export('abooks_validate.mp3', format='mp3')

with open('abooks_validate.raw', 'wb') as f:
    f.write(abooks_raw_data_validate)
