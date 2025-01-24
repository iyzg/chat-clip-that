import cv2
import pyaudio
import wave
import numpy as np
import threading
import time
from collections import deque
from datetime import datetime
import os
import subprocess
import pvporcupine
from pvrecorder import PvRecorder


class TimestampedBuffer:
    def __init__(self, maxlen):
        self.buffer = deque(maxlen=maxlen)
        self.lock = threading.Lock()

    def append(self, data):
        with self.lock:
            self.buffer.append((data, time.time()))

    def get_last_n_seconds(self, seconds, current_time=None):
        if current_time is None:
            current_time = time.time()
        with self.lock:
            return [
                (data, ts) for data, ts in self.buffer if current_time - ts <= seconds
            ]


def record_video(filename, frames_buffer):
    # Take only the last 5 seconds (150 frames) from the buffer
    frames_to_save = list(frames_buffer)[-150:]  # 30fps * 5 seconds

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(
        f"{filename}.avi",
        fourcc,
        30.0,
        (frames_to_save[0][0].shape[1], frames_to_save[0][0].shape[0]),
    )

    for frame, _ in frames_to_save:
        out.write(frame)
    out.release()


def record_audio(filename, duration):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100

    p = pyaudio.PyAudio()

    stream = p.open(
        format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK
    )

    frames = []

    # Record for specified duration
    for i in range(0, int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save the audio file
    wf = wave.open(f"{filename}.wav", "wb")
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b"".join(frames))
    wf.close()


def combine_audio_video(video_file, audio_file, output_file):
    try:
        # Use ffmpeg directly to combine audio and video
        cmd = [
            "ffmpeg",
            "-y",  # -y to overwrite output file if it exists
            "-i",
            video_file,  # video input
            "-i",
            audio_file,  # audio input
            "-c:v",
            "copy",  # copy video codec
            "-c:a",
            "aac",  # convert audio to aac
            "-strict",
            "experimental",
            output_file,
        ]

        subprocess.run(cmd, check=True, capture_output=True)

    except subprocess.CalledProcessError as e:
        print(f"Error combining files: {e.stderr.decode()}")
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise


def record_video_buffer(video_buffer, cap):
    """Continuously capture video frames with timestamps"""
    while True:
        ret, frame = cap.read()
        if ret:
            video_buffer.append(frame)


def record_audio_buffer(audio_buffer):
    """Continuously capture audio with timestamps"""
    CHUNK = 2048  # Increased buffer size
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100

    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
        input_device_index=None,  # Use default input device
        stream_callback=None,
    )

    try:
        while True:
            try:
                data = stream.read(
                    CHUNK, exception_on_overflow=False
                )  # Added overflow handling
                audio_buffer.append(data)
            except OSError as e:
                print(f"Audio buffer overflow, continuing... ({e})")
                continue
    finally:
        try:
            if stream.is_active():
                stream.stop_stream()
            stream.close()
        except:
            pass
        p.terminate()


def save_synchronized_recording(video_buffer, audio_buffer, base_filename, duration=15):
    """Save synchronized video and audio from buffers"""
    print("Starting to save synchronized recording...")
    current_time = time.time()

    try:
        # Get synchronized data from buffers
        video_data = video_buffer.get_last_n_seconds(duration, current_time)
        audio_data = audio_buffer.get_last_n_seconds(duration, current_time)

        # Save video with exact frame rate
        if video_data:
            print("Processing video...")
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            fps = 30.0
            frame_count = len(video_data)

            if frame_count >= 2:
                start_time = video_data[0][1]
                end_time = video_data[-1][1]
                actual_fps = (frame_count - 1) / (end_time - start_time)
                fps = actual_fps

            out = None
            try:
                out = cv2.VideoWriter(
                    f"{base_filename}.avi",
                    fourcc,
                    fps,
                    (video_data[0][0].shape[1], video_data[0][0].shape[0]),
                )

                sorted_frames = sorted(video_data, key=lambda x: x[1])
                for frame, _ in sorted_frames:
                    out.write(frame)
            finally:
                if out is not None:
                    out.release()
            print("Video processing complete")

        # Save audio with precise timing
        if audio_data:
            print("Processing audio...")
            wf = None
            try:
                wf = wave.open(f"{base_filename}.wav", "wb")
                wf.setnchannels(1)
                wf.setsampwidth(pyaudio.PyAudio().get_sample_size(pyaudio.paInt16))
                wf.setframerate(44100)

                sorted_audio = sorted(audio_data, key=lambda x: x[1])
                audio_chunks = [chunk for chunk, _ in sorted_audio]
                wf.writeframes(b"".join(audio_chunks))
            finally:
                if wf is not None:
                    wf.close()
            print("Audio processing complete")

    except Exception as e:
        print(f"Error during recording save: {e}")
        raise


def listen_for_trigger(callback):
    """Continuously listen for wake word using Porcupine"""
    porcupine = pvporcupine.create(
        access_key="nlkAazzcxIg6errNVqQPGx4AiZO7OHJcjCjpHtNUUWcsytjibpi6EQ==",
        keyword_paths=["keyword.ppn"],
    )

    recorder = PvRecorder(
        device_index=-1, frame_length=porcupine.frame_length  # default microphone
    )
    recorder.start()

    print(f"Listening for wake word... (using device: {recorder.selected_device})")

    try:
        while True:
            pcm = recorder.read()
            result = porcupine.process(pcm)
            if result >= 0:
                print("🎤 Wake word detected!")
                callback()

    except Exception as e:
        print(f"❌ Error in wake word detection: {e}")
    finally:
        recorder.delete()
        porcupine.delete()


def main():
    cap = cv2.VideoCapture(0)
    fps = 30.0
    buffer_seconds = 30
    save_duration = 15

    try:
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, fps)

        # Initialize buffers
        video_buffer = TimestampedBuffer(maxlen=int(fps * buffer_seconds))
        audio_buffer = TimestampedBuffer(maxlen=int(44100 * buffer_seconds / 1024))

        # Start capture threads
        video_thread = threading.Thread(
            target=record_video_buffer, args=(video_buffer, cap), daemon=True
        )
        audio_thread = threading.Thread(
            target=record_audio_buffer, args=(audio_buffer,), daemon=True
        )

        video_thread.start()
        audio_thread.start()

        def save_recording():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"recording_{timestamp}"

            try:
                save_synchronized_recording(
                    video_buffer, audio_buffer, base_filename, save_duration
                )
                combine_audio_video(
                    f"{base_filename}.avi",
                    f"{base_filename}.wav",
                    f"{base_filename}_final.mp4",
                )

                # Clean up intermediate files
                if os.path.exists(f"{base_filename}.avi"):
                    os.remove(f"{base_filename}.avi")
                if os.path.exists(f"{base_filename}.wav"):
                    os.remove(f"{base_filename}.wav")

                print(
                    f"✨ Successfully saved {save_duration} seconds as {base_filename}_final.mp4\n"
                )
            except Exception as e:
                print(f"Error during save process: {e}")

        # Start wake word detection in separate thread
        voice_thread = threading.Thread(
            target=listen_for_trigger, args=(save_recording,), daemon=True
        )
        voice_thread.start()

        print("Say wake word to save the last 5 seconds, press 'q' to quit")

        while True:
            frame = cap.read()[1]
            cv2.imshow("Recording Buffer", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
