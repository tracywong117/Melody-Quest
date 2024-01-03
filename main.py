import pygame as pg
import tkinter as tk
from tkinter import filedialog
import pygame.display
import threading
import time

import sys
import os
import os.path

import librosa
import wave
import libfmp.b
import libfmp.c3
import libfmp.c8

import numpy as np
import pandas as pd
import random
import joblib

from scipy.fftpack import dct
from scipy.interpolate import interp1d

def separate_melody_accompaniment(x, Fs, N, H, traj, n_harmonics=10, tol_cent=50.0):
    # Compute STFT
    X = librosa.stft(x, n_fft=N, hop_length=H, win_length=N, pad_mode='constant')
    Fs_feature = Fs / H
    T_coef = np.arange(X.shape[1]) / Fs_feature
    freq_res = Fs / N
    F_coef = np.arange(X.shape[0]) * freq_res

    # Adjust trajectory
    traj_X_values = interp1d(traj[:, 0], traj[:, 1], kind='nearest', fill_value='extrapolate')(T_coef)
    traj_X = np.hstack((T_coef[:, None], traj_X_values[:, None, ]))

    # Compute binary masks
    mask_mel = libfmp.c8.convert_trajectory_to_mask_cent(traj_X, F_coef, n_harmonics=n_harmonics, tol_cent=tol_cent)
    mask_acc = np.ones(mask_mel.shape) - mask_mel

    # Compute masked STFTs
    X_mel = X * mask_mel
    X_acc = X * mask_acc

    # Reconstruct signals
    x_mel = librosa.istft(X_mel, hop_length=H, win_length=N, window='hann', center=True, length=x.size)
    x_acc = librosa.istft(X_acc, hop_length=H, win_length=N, window='hann', center=True, length=x.size)

    return x_mel, x_acc

def random_discard(onset_times, onset_frames, threshold=1):
    i = 1
    new_onset_times = [onset_times[0]]
    new_onset_frames = [onset_frames[0]]
    while i < len(onset_times):
        diff = onset_times[i] - onset_times[i-1]
        if not diff < threshold or not random.random()>=0.5:
            new_onset_times.append(onset_times[i])
            new_onset_frames.append(onset_frames[i])
        i += 1
    return new_onset_times, new_onset_frames

# generation of beatmap
def generate_beatmap(file_path, melody_extract_flag=False):
    file_name = os.path.splitext(file_path)[0]
    if os.path.isfile(file_name + '_beatmap.txt'):
        print(f"The beatmap file {file_path} already exists.")
    else:
        y, sr = librosa.load(file_path)
        duration = librosa.get_duration(y=y, sr=sr)
        print(f'duration: {duration}')
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr, 
                                                  hop_length=100,
                                                  backtrack=False,
                                                  units='samples',
                                                #   pre_max=20,
                                                #   post_max=20,
                                                #   pre_avg=100,
                                                #   post_avg=100,
                                                #   delta=0.2,
                                                #   wait=0
                                                  )
        onset_times = librosa.samples_to_time(onset_frames)
        # onset_times = librosa.frames_to_time(onset_frames)
        onset_times -= 0.6 # time for the note to fall down
        onset_times = [x for x in onset_times if x >= 0]
        notes_num = len(onset_times)
        print(f'number of notes: {notes_num}')

        # calculate average rate
        avg_rate = notes_num/duration
        
        print(avg_rate)

        if avg_rate>3 and melody_extract_flag:
            # perform melody extraction and separation
            # however, this is too slow
            # Compute trajectory
            traj, Z, T_coef, F_coef_hertz, F_coef_cents = libfmp.c8.compute_traj_from_audio(y, Fs=sr, 
                                                    constraint_region=None, gamma=0.1)

            N = 2048
            H = N//4
            x_mel, x_acc = separate_melody_accompaniment(y, Fs=sr, N=N, H=H, traj=traj, n_harmonics=30, tol_cent=50)

            onset_frames = librosa.onset.onset_detect(y=x_mel, sr=sr)
            onset_times = librosa.frames_to_time(onset_frames)
            onset_times -= 0.6 # time for the note to fall down
            onset_times = [x for x in onset_times if x >= 0]
            new_notes_num = len(onset_times)
            print(f'number of notes after melody extraction: {new_notes_num}')
            if new_notes_num > notes_num:
                onset_times = random_discard(onset_times)
                print(f'number of notes after random discard: {len(onset_times)}')
        elif avg_rate>3 and not melody_extract_flag:
            print(onset_times)
            onset_times, onset_frames = random_discard(onset_times, onset_frames)
            print(f'number of notes after random discard: {len(onset_times)}')

        def estimate_pitch(segment, sr, fmin=50.0, fmax=2000.0):
            # Compute autocorrelation of input segment.
            r = librosa.autocorrelate(segment)

            # Define lower and upper limits for the autocorrelation argmax.
            r[:int(sr/fmax)] = 0 # delete all value after upper limits
            r[int(sr/fmin):] = 0 # delete all value before lower limit
            
            # Find the location of maximum autocorrelation
            loc = r.argmax() 

            # Find the location of the maximum autocorrelation and determine the fundamental frequency
            f0 = float(sr/loc)
            return f0
        
        def estimate_pitch_by_onset(x, onset_samples, i, sr):
            n0 = onset_samples[i]
            n1 = onset_samples[i+1]
            f0 = estimate_pitch(x[n0:n1], sr=sr)
            return f0
        
        y_pitch = np.array([
            estimate_pitch_by_onset(y, onset_frames, i, sr=sr)
            for i in range(len(onset_frames)-1)
        ])

        print(f'number of pitch: {len(y_pitch)}')
        print(f'number of onset_times: {len(onset_times)}')
        print(f'number of onset_frames: {len(onset_frames)}')

        min_pitch = np.min(y_pitch)
        max_pitch = np.max(y_pitch)
        sd_pitch = np.std(y_pitch)
        interval_pitch = (max_pitch - min_pitch) / 4.0
        range_pitch = [min_pitch, min_pitch+interval_pitch, min_pitch+interval_pitch*2, min_pitch+interval_pitch*3]
                        
        # export to .txt file
        # with open(file_name + '.txt', 'w') as f:
        #     f.write('\n'.join(['%.4f' % onset_time for onset_time in onset_times]))

        def diff(a,b):
            return abs(a-b)
        
        def between(num, this):
            for temp in range(len(this)-1):
                if num >= this[temp] and num <= this[temp+1]:
                    return temp
            return len(this)-1
        
        
        with open(file_name + '_beatmap.txt', 'w') as f:
            cur_col_index = between(y_pitch[0], range_pitch)
            ref_pitch = y_pitch[0]
            for i in range(len(onset_times)-1):
                if y_pitch[i] >= ref_pitch:
                    cur_col_index = int(cur_col_index + int(diff(y_pitch[i], ref_pitch)/(sd_pitch/3)))
                else:
                    cur_col_index = int(cur_col_index - int(diff(y_pitch[i], ref_pitch)/(sd_pitch/3)))
                ref_pitch = y_pitch[i]
                if cur_col_index<0 or cur_col_index>3:
                    cur_col_index = between(y_pitch[0], range_pitch)
                print(cur_col_index)
                f.write(f'{onset_times[i]:.4f}, {str(key_list[cur_col_index])} \n')


# detect the mood and return the color of mood
def get_color_from_mood_detection(file_path):
    def extract_features(audio_file):
        y, sr = librosa.load(audio_file)
        
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_stft_mean = np.mean(chroma_stft)
        chroma_stft_var = np.var(chroma_stft)
        
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_centroid_mean = np.mean(spectral_centroid)
        spectral_centroid_var = np.var(spectral_centroid)
        
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spectral_bandwidth_mean = np.mean(spectral_bandwidth)
        spectral_bandwidth_var = np.var(spectral_bandwidth)
        
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        spectral_contrast_mean = np.mean(spectral_contrast)
        spectral_contrast_var = np.var(spectral_contrast)
        
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
        zero_crossing_rate_mean = np.mean(zero_crossing_rate)
        zero_crossing_rate_var = np.var(zero_crossing_rate)
        
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_var = np.var(mfccs, axis=1)
        
        features = {
            'tempo': tempo.reshape(1,)[0],
            'beat_frames': beat_frames.mean(axis=0),
            'chroma_stft_mean': chroma_stft_mean,
            'chroma_stft_var': chroma_stft_var,
            'zero_crossing_rate_mean': zero_crossing_rate_mean,
            'zero_crossing_rate_var': zero_crossing_rate_var,
            'spectral_centroid_mean': spectral_centroid_mean,
            'spectral_centroid_var': spectral_centroid_var,
            'spectral_bandwidth_mean': spectral_bandwidth_mean,
            'spectral_bandwidth_var': spectral_bandwidth_var,
            'spectral_contrast_mean': spectral_contrast_mean,
            'spectral_contrast_var': spectral_contrast_var,
            'mfcc1_mean': mfccs_mean[0],
            'mfcc1_var': mfccs_var[0],
            'mfcc2_mean': mfccs_mean[1],
            'mfcc2_var': mfccs_var[1],
            'mfcc3_mean': mfccs_mean[2],
            'mfcc3_var': mfccs_var[2],
            'mfcc4_mean': mfccs_mean[3],
            'mfcc4_var': mfccs_var[3],
            'mfcc5_mean': mfccs_mean[4],
            'mfcc5_var': mfccs_var[4],
            'mfcc6_mean': mfccs_mean[5],
            'mfcc6_var': mfccs_var[5],
            'mfcc7_mean': mfccs_mean[6],
            'mfcc7_var': mfccs_var[6],
            'mfcc8_mean': mfccs_mean[7],
            'mfcc8_var': mfccs_var[7],
            'mfcc9_mean': mfccs_mean[8],
            'mfcc9_var': mfccs_var[8],
            'mfcc10_mean': mfccs_mean[9],
            'mfcc10_var': mfccs_var[9],
            'mfcc11_mean': mfccs_mean[10],
            'mfcc11_var': mfccs_var[10],
            'mfcc12_mean': mfccs_mean[11],
            'mfcc12_var': mfccs_var[11],
            'mfcc13_mean': mfccs_mean[12],
            'mfcc13_var': mfccs_var[12],
            'mfcc14_mean': mfccs_mean[13],
            'mfcc14_var': mfccs_mean[13],
            'mfcc15_mean': mfccs_mean[14],
            'mfcc15_var': mfccs_mean[14],
            'mfcc16_mean': mfccs_mean[15],
            'mfcc16_var': mfccs_mean[15],
            'mfcc17_mean': mfccs_mean[16],
            'mfcc17_var': mfccs_mean[16],
            'mfcc18_mean': mfccs_mean[17],
            'mfcc18_var': mfccs_mean[17],
            'mfcc19_mean': mfccs_mean[18],
            'mfcc19_var': mfccs_mean[18],
            'mfcc20_mean': mfccs_mean[19],
            'mfcc20_var': mfccs_mean[19],
        }

        return features

    features = extract_features(file_path)

    features_unseen_song = [value for value in features.values()]
    print(features_unseen_song)

    # Load the model from disk
    loaded_model = joblib.load('LR_arousal.sav')

    # Use the loaded model to make predictions
    arousal_prediction = loaded_model.predict([features_unseen_song])

    print(arousal_prediction)

    # Load the model from disk
    loaded_model = joblib.load('RF_valence.sav')

    # Use the loaded model to make predictions
    valence_prediction = loaded_model.predict([features_unseen_song])

    print(valence_prediction)

    if arousal_prediction > 0 and valence_prediction > 0:
        mood_color = (253, 242, 204) # Yellow
    elif arousal_prediction > 0 and valence_prediction < 0:
        mood_color = (194, 170, 224) # Purple 
    elif arousal_prediction < 0 and valence_prediction > 0:
        mood_color = (106, 247, 209) # Green
    else: 
        mood_color = (117, 193, 240) # blue

    print(mood_color)

    return mood_color

#### init ####
TITLE = "Melody's Quest"
WIDTH = 512
HEIGHT = 512
HEIGHT2 = 256.5 # height of a bar
WIDTH2 = 88 # width of a bar
FPS = 60 # frame rate

# key_list = ['A','S','K','L']
key_list = [40,167,297,417]
N = 4 # The number of bars
notes = []
note_list = []

score = 0
combo = 0
count = 0 # current note count
speed = 5 # speed of the notes

# Load song file
file_path = filedialog.askopenfilename(title="Select song file", filetypes=(("wav audio files", "*.wav"), ("mp3 audio files", "*.mp3")))
print(file_path)
file_name = os.path.splitext(file_path)[0]
print(file_name)

f = wave.open(file_path, 'rb') 
frequency = f.getframerate()
params = f.getparams()
nchannels, sampwidth, framerate, nframes = params[:4]

str_data  = f.readframes(nframes) 
wave_data = np.frombuffer(str_data, dtype = np.short)
wave_data.shape = -1,2
wave_data = wave_data.T

y, sr = librosa.load(file_path)
duration = librosa.get_duration(y=y, sr=sr)

generate_beatmap(file_path)

plot_width = WIDTH
plot_height = HEIGHT - 300
plot_color = (218, 94, 124)
buffer_size = int(1 / FPS * sr) # buffer size for the waveform
bar_spacing = 2
playback_speed = 1.0

mood_color = get_color_from_mood_detection(file_path)

fpsclock = pygame.time.Clock()
current_time = 0
#### init ####

class Game:
    def __init__(self):
        # Initialize game state
        self.state = "intro"

        # Initialize Pygame
        pg.init()

        # Load fonts
        self.basic_font = pg.font.SysFont("Arial", 15) 
        self.title_font = pg.font.SysFont("Arial", 15)
        self.combo_font = pg.font.SysFont("Arial", 30)

        # Create the game screen
        self.screen = pg.display.set_mode((WIDTH, HEIGHT))

        # Load images
        self.start_background = pg.image.load("start_background.png")
        self.main_background = pg.image.load("main_background.png")

        # when pressing the key A, S, K, L, the bar will be highlighted
        self.bar = pg.image.load("a.png")
        self.barA = pg.image.load("a.png")
        self.barS = pg.image.load("s.png")
        self.barK = pg.image.load("k.png")
        self.barL = pg.image.load("l.png")

        self.grade = pg.image.load("perfect.png")
        self.perfect = pg.image.load("perfect.png")
        self.good = pg.image.load("good.png")
        self.poor = pg.image.load("poor.png")
        self.miss = pg.image.load("miss.png")
        
        self.noteA = pg.image.load("a-note.png")
        self.noteS = pg.image.load("s-note.png")
        self.noteK = pg.image.load("k-note.png")
        self.noteL = pg.image.load("l-note.png")

        self.endbar = pg.image.load("end_bar.png")
        self.scoreText = pg.image.load("Score.png")
        self.comboText = pg.image.load("Combo.png")

        self.value = False

        # Set game window caption
        pg.display.set_caption(TITLE)

        # Create a Pygame clock object
        self.clock = pg.time.Clock()

    def read_beatmap(self):
        with open(file_name+"_beatmap.txt", 'r') as f:
            onset_times = []
            onset_num = []
            for line in f:
                parts = line.strip().split(',')
                onset_times.append(float(parts[0]))
                onset_num.append(int(parts[1]))
        return onset_times, onset_num   

    def load_beatmap(self):
        self.onset_times, self.onset_nums = self.read_beatmap()
        self.num_notes = len(self.onset_times)
        self.notes = [self.onset_times, self.onset_nums]

    # return current playing time (seconds)
    def get_current_time(self):
        return max(0, pygame.mixer.music.get_pos() / 1000)
    
    def waveform_drawing(self):
        # Get current time
        current_time = pygame.time.get_ticks() / 1000.0 - offset_time
        # print(current_time)
        
        # Get audio data for the current time and buffer size
        start_sample = int(current_time * sr * playback_speed)
        audio_data = y[start_sample:start_sample + buffer_size]
        
        # Draw waveform plot
        pygame.draw.line(self.screen, mood_color, (0, plot_height//2), (plot_width, plot_height//2))
        for i in range(len(audio_data)-1):
            x1 = int(i * plot_width / len(audio_data))
            x2 = int((i+1) * plot_width / len(audio_data))
            y1 = int((audio_data[i] + 1) * plot_height / 2)
            y2 = int((audio_data[i+1] + 1) * plot_height / 2)
            pygame.draw.line(self.screen, mood_color, (x1, y1), (x2, y2), 2)

    def _draw_game_info(self, num):
        num = int(num)
        height_map = abs(dct(wave_data[0][nframes - num:nframes - num + N]))
        height_map = [min(HEIGHT2, int(i ** (1 / 2.5) * HEIGHT2 / 100)) for i in height_map]
        self.draw_notes(height_map)

    # Draw game information 
    def draw_game_info(self):
        self.num -= framerate / 60
        if self.num > 0:
            self._draw_game_info(self.num)
        
        self.screen.blit(self.scoreText, (207, 9))
        self.screen.blit(self.comboText, (15, 9))
        self.screen.blit(self.grade, (202, 130))
        self.screen.blit(self.score, (235, 66))
        self.screen.blit(self.combo, (21, 41))
        pygame.draw.rect(self.screen, mood_color, [0,0,448 *((self.get_current_time()/duration)),10])

        pygame.display.flip()

    def draw_notes(self, height_map):
        global notes
        global count
        global combo
        global speed

        bars = []
        notes = []
        
        for i in height_map:
            bars.append([len(bars) * WIDTH2,  768 - i, WIDTH2 - 1, i])
        
        for i in bars:
            note_list.append([i[0], 233, i[3]])
        
        for i in range(0, self.num_notes):
            if self.get_current_time() >= self.notes[0][i]:
                if note_list[i][1] <= 500:
                    notes.append(pygame.draw.rect(self.screen, [255,255,255], [self.notes[1][i] + 20, note_list[i][1], 20, 20], 5, 20, 20, 20))
                    self.screen.blit(self.noteA, (self.notes[1][i], note_list[i][1]))
                    note_list[i][1] += speed
                    
                    t = threading.Thread(target=self.handle_miss)
                    t.start()

        self.screen.blit(self.endbar, (0, 485))

    def handle_miss(self):
        global notes
        global combo
        if notes[0].top >= 494:
            if self.value == False:
                self.grade = self.miss
                combo = 0
                notes.pop(0)
                time.sleep(1000)
            else:
                self.value = False
                notes.pop(0)
                time.sleep(1000)

    def intro(self):
        # display the intro background image
        self.screen.blit(self.start_background, (0, 0))
        
        # handle events
        self.events()
        pg.display.flip()
    
    def main_game_init(self):
        self.a_rect = self.noteA.get_rect()
        self.a_rect.left = 22
        self.a_rect.top = 460
        self.a_rect.height = 40

        self.s_rect = self.noteS.get_rect()
        self.s_rect.left = 148.5
        self.s_rect.top = 460
        self.s_rect.height = 40

        self.k_rect = self.noteK.get_rect()
        self.k_rect.left = 279.5
        self.k_rect.top = 460
        self.k_rect.height = 40

        self.l_rect = self.noteL.get_rect()
        self.l_rect.left = 399.5
        self.l_rect.top = 460
        self.l_rect.height = 40

        pygame.mixer.init(frequency=frequency)
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        pygame.mixer.music.set_endevent()
        
        self.load_beatmap()
        
        self.main_game()

    def main_game(self):
        global score
        global combo
        global count
        global speed
        global duration
        global offset_time 

        offset_time = pygame.time.get_ticks() / 1000.0
        
        # game loop
        while True:
            # handle events
            self.events()
            self.screen.blit(self.main_background, (0, 0))

            # update game information
            self.score =  self.combo_font.render(str(score), True, (255, 255, 255))
            self.combo = self.combo_font.render(str(combo), True, (246, 193, 66))
            self.waveform_drawing()
            # Limit the frame rate
            fpsclock.tick(60)
            self.draw_game_info()

    
    def handle_score(self, rect, i):
        global notes
        global score
        global combo
        global count
        global speed

        if abs(rect.top - i.top) < 10:
            self.grade = self.perfect
            if combo >= 1:
                score = score + (1000 * combo)
            else:
                score = score + 1000
        elif abs(rect.top - i.top) <= 15:
            self.grade = self.good

            if combo >= 1:
                score = score + (500 * combo)
            else:
                score = score + 500
        else:
            self.grade = self.poor
            if combo >= 1:
                score = score + (300 * combo)
            else:
                score = score + 300
        notes.clear()
        self.value = True
        count = count +1
        combo = combo +1        


    def events(self):
        global notes
        global score
        global combo
        global count
        global speed

        for event in pg.event.get():
            if self.state == "intro" and event.type == pg.KEYDOWN:
                if event.key == pg.K_RETURN:
                    if self.state == 'intro':
                        self.state = 'main_game'
            
            if event.type == pg.QUIT:
                self.quit()
            elif self.state == "main_game" and event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                   self.quit()
                # when the user press a
                if event.key == pg.K_a:
                  #  self.beep.play()
                    self.bar = self.barA
                    self.screen.blit(self.bar, (22, 233.5))
                    for i in notes:
                        if(self.a_rect.colliderect(i)):
                            self.handle_score(self.a_rect, i)
                    pg.display.flip()
                # when the user press s
                if event.key == pg.K_s:
                    self.bar = self.barS
                    self.screen.blit(self.bar,(148.5, 233.5))
                    for i in notes:
                        if (self.s_rect.colliderect(i)):
                            self.handle_score(self.s_rect, i)
                    pg.display.flip()
                # when the user press k
                if event.key == pg.K_k:
                    self.bar = self.barK
                    self.screen.blit(self.bar,(279.5, 233.5))
                    for i in notes:
                        if (self.k_rect.colliderect(i)):
                            self.handle_score(self.k_rect, i)
                    pg.display.flip()
                # when the user press l
                if event.key == pg.K_l:
                    self.bar = self.barL
                    self.screen.blit(self.bar,(399.5, 233.5))
                    for i in notes:
                        if (self.l_rect.colliderect(i)):
                            self.handle_score(self.l_rect, i)
                    pg.display.flip()

    def handle_game_state(self):
        if self.state == 'intro':
            self.intro()

        if self.state == 'main_game':
            self.main_game_init()

    def quit(self):
        pg.quit()
        sys.exit()

# new game instance
my_game = Game()
my_game.num = nframes

# run game loop
while True:
    my_game.handle_game_state()
