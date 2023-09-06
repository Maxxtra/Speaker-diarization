#!/usr/bin/env python
# coding: utf-8

# 1. Introduction
# 
# This script is designed to process audio files in a specified folder, perform speaker diarization
# using the Pyannote library, and transcribe the audio segments using the Whisper ASR model. The
# transcribed data is then stored in a CSV file for further analysis.

# In[1]:


# !pip install -qq https://github.com/pyannote/pyannote-audio/archive/refs/heads/develop.zip
# !pip install -qq ipython==7.34.0
# !pip install -q git+https://github.com/openai/whisper.git
# !pip install torchaudio
# !pip install pydub
# !apt-get -qq install -y ffmpeg


# In[2]:


import os
import csv
import subprocess
from tqdm import tqdm
from typing import Tuple

import torch
import whisper
import pandas as pd

from pydub import AudioSegment
from whisper.audio import pad_or_trim, log_mel_spectrogram, N_FRAMES
from pyannote.audio import Pipeline


# 2. PyannoteProcessor
# 
# Class to perform speaker diarization using the Pyannote library.

# In[3]:


class PyannoteProcessor:
    """
    Class to perform speaker diarization using the Pyannote library.
    """

    def __init__(self):
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization",
            use_auth_token="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",
        )

    def perform_diarization(self, audio_file_path):
        self.pipeline.to(torch.device('cuda')) # switch to gpu
        diarization = self.pipeline(audio_file_path)

        with open ("sample.rttm", "w") as rttm:
          diarization.write_rttm(rttm)
        

    def rttm_to_dataframe(self, rttm_file_path):
        columns = [
            "Type",
            "File ID",
            "Channel",
            "Start Time",
            "Duration",
            "Orthography",
            "Confidence",
            "Speaker",
            "x",
            "y",
        ]
        data = []

        with open(rttm_file_path, "r") as rttm_file:
            lines = rttm_file.readlines()

        for line in lines:
            line = line.strip().split()
            data.append(line)

        df = pd.DataFrame(data, columns=columns)
        df = df.drop(["x", "y", "Orthography", "Confidence"], axis=1)
        return df


# 3. WhisperProcessor
# 
# Class to process audio segments using the Whisper ASR model.

# In[4]:


class WhisperProcessor:
    def __init__(self):
        # Initialize the Whisper model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(self.device)
        self.model = whisper.load_model("large-v2").to(self.device)

    def transcribe_audio_with_whisper(self, audio_file, detected_language):
        """
        Transcribes an audio segment using the Whisper ASR model.

        Args:
            audio_file (str): Path to the audio file.
            detected_language (str): Detected language of the audio.

        Returns:
            dict: Transcription result containing text and other information.
        """
        result = self.model.transcribe(
            audio_file, language=detected_language, fp16=False, temperature = (0.8, 1.0)
        )
        return result

    def detect_audio_language(self, audio) -> Tuple[str, float]:
        """
        Detects the language of an audio segment using the Whisper ASR model.

        Args:
            audio (AudioSegment): Audio segment to detect language from.

        Returns:
            Tuple[str, float]: Detected language and confidence.
        """
        mel_segment = pad_or_trim(log_mel_spectrogram(audio), N_FRAMES).to(
            self.model.device
        )
        _, probs = self.model.detect_language(mel_segment)
        detected_language = max(probs, key=probs.get)
        confidence = probs[detected_language]

        return detected_language, confidence

    def process_audio_segment(
        self, audio_file, start_time, end_time, detected_language
    ):
        """
        Processes an audio segment within a specified time range.

        Args:
            audio_file (str): Path to the audio file.
            start_time (int): Start time of the segment in milliseconds.
            end_time (int): End time of the segment in milliseconds.
            detected_language (str): Detected language of the audio segment.

        Returns:
            str: Collapsed transcript for the processed audio segment.
        """
        start_time = float(start_time * 1000)
        end_time = float(end_time * 1000)

        audio = AudioSegment.from_file(audio_file)
        audio_segment = audio[start_time:end_time]

        audio_segment_path = f"/root/diar/audio_segment_{start_time}.wav"
        audio_segment.export(audio_segment_path, format="wav")

        # Transcribe the audio segment
        transcription_result = self.transcribe_audio_with_whisper(
            audio_segment_path, detected_language
        )
        whisper_transcript = transcription_result["text"]

        # Split the transcript into segments
        segments = whisper_transcript.split("\n")

        # Collapse the segments
        collapsed_transcript = self.collapse_segments(segments)

        # Delete the temporary audio segment file
        os.remove(audio_segment_path)

        return collapsed_transcript

    def collapse_segments(self, transcript_segments):
        """
        Collapses individual words and spaces in the transcript segments.

        Args:
            transcript_segments (list): List of transcript segments.

        Returns:
            str: Collapsed transcript with words and spaces combined.
        """
        segment_counter = 0
        collapsed_segments = []

        for segment in transcript_segments:
            if segment.startswith("Segment"):
                segment_counter += 1
                collapsed_segments.append(segment)
            else:
                words = segment.split()
                for word in words:
                    collapsed_segments.append(word)
                    collapsed_segments.append(" ")

        collapsed_transcript = "".join(collapsed_segments)
        return collapsed_transcript


# 4. AudioProcessor
# 
# Class to process audio segments, transcribe using Whisper, and add to the CSV.

# In[5]:


class AudioProcessor:
    def __init__(self, pyannote_processor, whisper_processor):
        """
        Initialize the AudioProcessor class.

        Args:
            whisper_processor (WhisperProcessor): Instance of the WhisperProcessor class.
            pyannote_processor (PyannoteProcessor): Instance of the PyannoteProcessor class.
        """
        self.pyannote = pyannote_processor
        self.whisper = whisper_processor

    def process_and_append_to_csv(
        self, audio_file, detected_language, csv_writer, conv_number
    ):
        """
        Process audio segments, transcribe using Whisper, and add to the CSV.

        Args:
            audio_file (str): Path to the audio file.
            detected_language (str): Detected language of the audio.
            csv_writer (csv.writer): CSV writer object to write to the CSV.
            conv_number (int): Conversation number for the CSV.

        Returns:
            None
        """
        print("Starting Pyannote...")
        self.pyannote.perform_diarization(audio_file)
        print("Finished Pyannote")

        rttm_file_path = "sample.rttm"
        df = self.pyannote.rttm_to_dataframe(rttm_file_path)
        df = df.astype({"Start Time": "float"})
        df = df.astype({"Duration": "float"})
        df["Utterence"] = None
        df["End Time"] = df["Start Time"] + df["Duration"]

        silence_gap_pairs = []

        for ind in df.index:
            start_time = df["Start Time"][ind]
            end_time = df["End Time"][ind]
            speaker = df["Speaker"][ind]

            silence_gap_pairs.append((start_time, end_time, speaker))

        attributes_list = []

#         for i in range(len(silence_gap_pairs) - 1):
#             current_start, current_end, current_speaker = silence_gap_pairs[i]
#             next_start, next_end, next_speaker = silence_gap_pairs[i + 1]

#             if current_speaker == next_speaker:
#                 attributes_list.append((current_start, next_end, current_speaker, ""))
#             if current_speaker != last_speaker:
#                 attributes_list.append(
#                     (current_start, current_end, current_speaker, "")
#                 )

#             last_speaker = current_speaker
        current_start, current_end, current_speaker = silence_gap_pairs[0]

        for start, end, speaker in silence_gap_pairs[1:]:
            if speaker == current_speaker:
                current_end = end
            else:
                attributes_list.append((current_start, current_end, current_speaker, ""))
                current_start, current_end, current_speaker = start, end, speaker

        print("Starting whisper...")
        for i, (start, end, speaker, text) in enumerate(attributes_list):
            transcript = self.whisper.process_audio_segment(
                audio_file, start, end, detected_language
            )
            attributes_list[i] = (start, end, speaker, transcript)
            
        print("Finished whisper")
        
        for start, end, speaker, text in attributes_list:
            csv_writer.writerow([conv_number, audio_file[:-4], speaker, text])
    

    def process_folder(self, folder_path, output_csv_path):
        """
        Process audio files in a folder in ascending order of dimension
        and create a CSV file with transcriptions.

        Args:
            folder_path (str): Path to the folder containing WAV files.

        Returns:
            None
        """
        checkpoint_interval = 10
        checkpoint_data = []

        # Get a list of WAV files in the folder
        wav_files = [
            filename
            for filename in os.listdir(folder_path)
            if filename.endswith(".wav")
        ]

        # Function to get the duration of a WAV file
        def get_wav_duration(file_path):
            command = ['soxi', '-D', file_path]
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode == 0:
                return float(result.stdout.strip())
            else:
                return float('inf')  # Return a very large value for files with errors

        # Sort the WAV files based on their duration
        wav_files.sort(key=lambda file: get_wav_duration(os.path.join(folder_path, file)))

        # Determine the highest conv_id already processed
        last_conv_id = 0
        if os.path.exists(output_csv_path):
            with open(output_csv_path, mode="r", encoding="utf-8") as csv_file:
                csv_reader = csv.reader(csv_file)
                next(csv_reader)  # Skip header
                for row in csv_reader:
                    conv_id = int(row[1])  # Assuming conv_id is in the second column
                    last_conv_id = max(last_conv_id, conv_id)

        with open(output_csv_path, mode="a", newline="", encoding="utf-8") as csv_file:
            csv_writer = csv.writer(
                csv_file, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )

            try:
                for conv_number, audio_file in enumerate(tqdm(wav_files, desc="Processing audio files")):
                    # Skip files that were already processed
                    if conv_number < last_conv_id:
                        continue

                    audio_file_path = os.path.join(folder_path, audio_file)
                    duration = get_wav_duration(audio_file_path)

                    if duration is None:
                        print(f"Skipping {audio_file} due to error in duration retrieval")
                        continue

                    if duration == 0.0:
                        print(f"Skipping {audio_file} as it is an empty file")
                        continue

                    print(f"Processing {audio_file}...")
                    detected_language, confidence = self.whisper.detect_audio_language(
                        audio_file_path
                    )

                    self.process_and_append_to_csv(
                        audio_file_path, detected_language, csv_writer, conv_number
                    )
                    print(f"Finished processing {audio_file}")

                    # Save progress every checkpoint_interval iterations
                    if (conv_number + 1) % checkpoint_interval == 0:
                        csv_file.flush()  # Flush the buffer to the file
                        print(f"Saved progress at file {conv_number + 1}")

            except Exception as e:
                print(f"An error occurred: {e}")
                print("Saving progress before exiting...")
                csv_file.flush()  # Flush the buffer to the file
                print("Progress saved.")


# 5. Main Function

# In[6]:


def main():
    diarization = PyannoteProcessor()
    stt = WhisperProcessor()
    audio_processor = AudioProcessor(diarization, stt)

    # Input audio folder path
    folder_path = input("Enter the path to the folder containing WAV files: ")
    output_path = input("Enter the path where to same transcripts.csv file: ")
    output_csv_path = os.path.join(output_path, "transcripts.csv")
    audio_processor.process_folder(folder_path, output_csv_path)


if __name__ == "__main__":
    torch.cuda.empty_cache()
    get_ipython().system('gpustat')
    main()
    print("done")

