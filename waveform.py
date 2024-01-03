import pygame
import librosa
import numpy as np

#music_file = "short-suzume.wav"
music_file = "La Campanella.wav"

# Load the music file
y, sr = librosa.load(music_file)

# Initialize Pygame
pygame.init()

# Set the window dimensions
WIDTH, HEIGHT = 800, 400
screen = pygame.display.set_mode((WIDTH, HEIGHT))

plot_width = WIDTH
plot_height = HEIGHT - 100
plot_color = (218, 94, 124)

# Frame rate
FPS = 60

# Set the buffer size for the waveform
buffer_size = int(1 / FPS * sr)

# Set the bar spacing for the waveform
bar_spacing = 2

# Set the current time to zero
current_time = 0

# Set the playback speed of the music
playback_speed = 1.0

# Set the volume of the music
volume = 1.0

# Set the Pygame mixer frequency and buffer size
pygame.mixer.init(frequency=sr, buffer=buffer_size)

# Load the music into the Pygame mixer
pygame.mixer.music.load(music_file)

# Start playing the music
pygame.mixer.music.play()

fpsclock = pygame.time.Clock()

# Main loop
while True:
    # Get the current time
    current_time = pygame.time.get_ticks() / 1000.0
    
    # Get the audio data for the current time and buffer size
    start_sample = int(current_time * sr * playback_speed)
    audio_data = y[start_sample:start_sample + buffer_size]
    
    # Clear the screen
    screen.fill((255, 255, 255))
    
    # Draw the waveform plot
    pygame.draw.line(screen, plot_color, (0, plot_height//2), (plot_width, plot_height//2))
    for i in range(len(audio_data)-1):
        x1 = int(i * plot_width / len(audio_data))
        x2 = int((i+1) * plot_width / len(audio_data))
        y1 = int((audio_data[i] + 1) * plot_height / 2)
        y2 = int((audio_data[i+1] + 1) * plot_height / 2)
        pygame.draw.line(screen, plot_color, (x1, y1), (x2, y2), 2)
    
    # Update the Pygame screen
    #pygame.display.update()
    
    # Check for Pygame events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                pygame.quit()
                quit()
            elif event.key == pygame.K_UP:
                playback_speed += 0.1
                pygame.mixer.music.set_pos(current_time * playback_speed)
            elif event.key == pygame.K_DOWN:
                playback_speed -= 0.1
                pygame.mixer.music.set_pos(current_time * playback_speed)
            elif event.key == pygame.K_LEFT:
                volume -= 0.1
                pygame.mixer.music.set_volume(volume)
            elif event.key == pygame.K_RIGHT:
                volume += 0.1
                pygame.mixer.music.set_volume(volume)

    fpsclock.tick(FPS)