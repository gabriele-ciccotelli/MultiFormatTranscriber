# Whisper Queue

## Description

Whisper Queue is a program designed to simplify the process of transcribing audio and video files using the Whisper transcription model, owned by OpenAI.
The added value of this program lies in the fact that it creates a workflow whereby, if the user has to transcribe several audio and video files, even those with extensions not supported by the Whisper model, the program will automatically start each transcription, one after the other, using computing resources and time more efficiently, and will attempt to convert files not originally supported by Whisper, into a format supported by the model.
The program was born from my need to transcribe long university lectures in a specific order to be able to prepare for an exam in time. It often happened that, after starting a transcription, it would end in the middle of the night, so the computer would remain switched on without performing any useful tasks until the next morning, when I would start a new transcription. To avoid this waste of electricity and inefficient use of computational resources, I wrote this little program that I decided to make public in case anyone else was in the same situation as me.
So, Whisper Queue is particularly useful for those who have to transcribe large amounts of files in different formats.


## Requirements

To run Whisper Transcriber, you need the following Python packages installed:

- [moviepy](https://zulko.github.io/moviepy/) - distributed under the MIT licence.
- [whisper](https://github.com/openai/whisper) - distributed under the MIT licence.
- [torch](https://github.com/pytorch/pytorch) - distributed under the BSD-style licence.

You can install them using pip:

```bash
pip install moviepy whisper torch
```

Additionally, the program requires ffmpeg for media file conversion. You can install ffmpeg by following this command:

```bash
pip install python-ffmpeg
```

## Usage
Run the program using Python:

```bash
python whisper_transcriber.py
```

The program will guide you through a series of prompts to configure the transcription:

1. Device Choice: Choose whether to use the GPU (if CUDA compatible) or the CPU for transcription.
2. Model Choice: Select the Whisper model to use (tiny, base, small, medium, large-v2, large-v3).
3. File/Folder Path: Enter the path to the file to be transcribed or the folder containing all the files to be transcribed.
4. Language: Specify the language to usr for the transcription.
5. Output Path: Enter the path to the folder where you want to save the single transcription or all the transcriptions.
6. Order Criterion: If you chose a folder with multiple files, select the order criterion for transcription (by creation date, modification date, custom order, etc.).

In case you want to use a customised file transcription order, all you have to do is rename the files by entering the numbers inside round brackets.
E.g., "file_name(1).mp3", "(1)file_name.mp3", etc.
