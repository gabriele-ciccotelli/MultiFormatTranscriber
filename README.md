# MultiFormatTranscriber

## Purpose

A tool for transcribing audio and video files in multiple formats. Automatically converts unsupported formats for compatibility with Whisper.

## Description

MultiFormatTranscriber is a program designed to simplify the process of transcribing audio and video files using the Whisper transcription model, owned by OpenAI.
The added value of this program lies in the fact that it creates a workflow whereby, if the user has to transcribe several audio and video files, even those with extensions not supported by the Whisper model, the program will automatically start each transcription, one after the other, using computing resources and time more efficiently, and will attempt to convert files not originally supported by Whisper into a format supported by the model.
The program was born from my need to transcribe long university lectures in a specific order to be able to prepare for an exam in time. It often happened that, after starting a transcription, it would end in the middle of the night, so the computer would remain switched on without performing any useful tasks until the next morning, when I would start a new transcription. To avoid this waste of electricity and inefficient use of computational resources, I wrote this little program that I decided to make public in case anyone else was in the same situation as me.
So, MultiFormatTranscriber is particularly useful for those who have to transcribe large amounts of files in different formats.

## Requirements

To run MultiFormatTranscriber, you need the following Python packages installed:

- [whisper](https://github.com/openai/whisper) - distributed under the MIT licence.
- [torch](https://github.com/pytorch/pytorch) - distributed under the BSD-style licence.

You can install them using pip:

```bash
pip install whisper torch
```

Additionally, the program requires FFmpeg for media file conversion.

### Installing and Configuring FFmpeg

FFmpeg is a crucial component for handling various audio/video formats and converting them when necessary. You need to install it separately and ensure it's accessible from your system's command line (i.e., it's in your system's PATH).

1. Download FFmpeg  
   Go to the official FFmpeg download page: https://ffmpeg.org/download.html  
   Download the version appropriate for your operating system (Windows, macOS, Linux). For Windows, you'll typically download a .zip file. For macOS and Linux, you might prefer using a package manager.

2. Install FFmpeg and Add to PATH

**Windows**  
1. Extract the downloaded .zip file to a folder on your computer (e.g., `C:\FFmpeg`).  
2. You need to add the `bin` subfolder (e.g., `C:\FFmpeg\bin`) to your system's PATH environment variable:  
   - Search for "environment variables" in the Windows Start Menu and select "Edit the system environment variables".  
   - In the System Properties window, click the "Environment Variables..." button.  
   - Under "System variables" (or "User variables"), find the variable named `Path` and select it.  
   - Click "Edit...", then "New", and paste the full path to the FFmpeg `bin` folder (e.g., `C:\FFmpeg\bin`).  
   - Click "OK" on all open windows to save the changes.  
3. You might need to restart your command prompt or your computer for the changes to take effect.

**macOS**  
- **Using Homebrew (recommended)**:  
  ```bash
  brew install ffmpeg
  ```  
  Homebrew usually handles adding it to your PATH automatically.  
- **Manual Installation**:  
  If you download the FFmpeg package manually, extract it, and then add the `bin` directory to your PATH by editing your shell's configuration file (e.g., `~/.zshrc` or `~/.bash_profile`):
  ```bash
  export PATH="/path/to/your/ffmpeg/bin:$PATH"
  ```  
  Replace `/path/to/your/ffmpeg/bin` with the actual path. Then run `source ~/.zshrc` or open a new terminal window.

**Linux**  
- **Using a Package Manager (recommended)**:  
  - Debian/Ubuntu:  
    ```bash
    sudo apt update && sudo apt install ffmpeg
    ```  
  - Fedora:  
    ```bash
    sudo dnf install ffmpeg
    ```  
  - Arch Linux:  
    ```bash
    sudo pacman -S ffmpeg
    ```  
  These commands usually add FFmpeg to your PATH automatically.  
- **Manual Installation**:  
  If you download a precompiled binary or compile from source, you need to add its `bin` directory to your PATH by editing your shell's configuration file (e.g., `~/.bashrc`, `~/.zshrc`):
  ```bash
  export PATH="/path/to/your/ffmpeg/bin:$PATH"
  ```  
  Then run `source ~/.bashrc` or open a new terminal window.

3. Verify Installation  
Open a new command prompt (Windows) or terminal (macOS/Linux), type:
```bash
ffmpeg -version
```
and press Enter. If FFmpeg is installed correctly and in your PATH, you should see version information; otherwise, recheck your PATH configuration.

## Usage

Run the program using Python:

```bash
python transcriber.py
```

The program will guide you through a series of prompts to configure the transcription:

1. Device Choice: Choose whether to use the GPU (if CUDA compatible) or the CPU for transcription.  
2. Model Choice: Select the Whisper model to use (tiny, base, small, medium, large-v2, large-v3).  
3. File/Folder Path: Enter the path to the file to be transcribed or the folder containing all the files to be transcribed.  
4. Language: Specify the language to use for the transcription.  
5. Output Path: Enter the path to the folder where you want to save the single transcription or all the transcriptions.  
6. Order Criterion: If you chose a folder with multiple files, select the order criterion for transcription (by creation date, modification date, custom order, etc.).

In case you want to use a customised file transcription order, all you have to do is rename the files by entering the numbers inside round brackets.  
E.g., `file_name(1).mp3`, `(1)file_name.mp3`, etc.