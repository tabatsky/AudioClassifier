import os
import math
import random

from pydub import AudioSegment

FRAME_RATES = [8000, 16000, 32000]
SAMPLE_LENGTH = 24000

artist_count = 3

samples_per_frame_rate_per_file = 200
samples_per_file = len(FRAME_RATES) * samples_per_frame_rate_per_file

files_per_artist_train = 12
files_per_artist_validate = 4

files_per_artist_total = files_per_artist_train + files_per_artist_validate

count_per_artist_train = files_per_artist_train * samples_per_file
count_per_artist_validate = files_per_artist_validate * samples_per_file

rnd = random.Random()

artist_raw_data_train = bytes(0)
artist_raw_data_validate = bytes(0)

artist = 0
while artist < artist_count:
    artist += 1

    artist_filenames = os.listdir(f'artist{artist}_train')
    print(len(artist_filenames), artist_filenames)

    for n in range(files_per_artist_train):
        fn_in = f'artist{artist}_train/{artist_filenames[n]}'
        sound = AudioSegment.from_mp3(fn_in)

        for frame_rate in FRAME_RATES:
            print(n, fn_in, sound.frame_rate, sound.sample_width, sound.channels)
            print(len(sound.raw_data), len(sound.raw_data) / (sound.frame_rate * sound.sample_width * sound.channels))
            sound = sound.set_frame_rate(frame_rate)
            sound = sound.set_channels(1)
            sound = sound.set_sample_width(1)
            print(len(sound.raw_data), len(sound.raw_data) / frame_rate)

            total_len = len(sound.raw_data)

            i = 0
            while i < samples_per_frame_rate_per_file:
                start = 0.9 * rnd.random() + 0.05
                start = round(start * total_len)
                if total_len - start < 3 * SAMPLE_LENGTH:
                    continue
                chunk_raw_data = sound.raw_data[start:start + SAMPLE_LENGTH]
                artist_raw_data_train += chunk_raw_data
                i += 1

    for k in range(files_per_artist_validate):
        n = k + files_per_artist_train
        fn_in = f'artist{artist}_train/{artist_filenames[n]}'
        sound = AudioSegment.from_mp3(fn_in)

        for frame_rate in FRAME_RATES:
            print(n, fn_in, sound.frame_rate, sound.sample_width, sound.channels)
            print(len(sound.raw_data), len(sound.raw_data) / (sound.frame_rate * sound.sample_width * sound.channels))
            sound = sound.set_frame_rate(frame_rate)
            sound = sound.set_channels(1)
            sound = sound.set_sample_width(1)
            print(len(sound.raw_data), len(sound.raw_data) / frame_rate)

            total_len = len(sound.raw_data)

            i = 0
            while i < samples_per_frame_rate_per_file:
                start = 0.9 * rnd.random() + 0.05
                start = round(start * total_len)
                if total_len - start < 3 * SAMPLE_LENGTH:
                    continue
                chunk_raw_data = sound.raw_data[start:start + SAMPLE_LENGTH]
                artist_raw_data_validate += chunk_raw_data
                i += 1

artist_raw_sound_train = AudioSegment(artist_raw_data_train, frame_rate=FRAME_RATES[0], sample_width=1, channels=1)

# artist_raw_sound_train.export('artist_train.pcm', format='u8')
# artist_raw_sound_train.export('artist_train.mp3', format='mp3')

with open(f'artist_{artist_count}_{samples_per_file}_{files_per_artist_total}_train.raw', 'wb') as f:
    f.write(artist_raw_data_train)

artist_raw_sound_validate = AudioSegment(artist_raw_data_validate, frame_rate=FRAME_RATES[0], sample_width=1, channels=1)

# artist_raw_sound_validate.export('artist_validate.pcm', format='u8')
# artist_raw_sound_validate.export('artist_validate.mp3', format='mp3')

with open(f'artist_{artist_count}_{samples_per_file}_{files_per_artist_total}_validate.raw', 'wb') as f:
    f.write(artist_raw_data_validate)

