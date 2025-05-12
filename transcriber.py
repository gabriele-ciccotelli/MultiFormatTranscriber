import os
import re
import torch
import whisper
import subprocess
from datetime import datetime
from typing import List, Optional
from whisper.model import Whisper as WhisperModel


# Supported extensions
EXTENSIONS_SUPPORTED: List[str] = [".mp3", ".mp4", ".mpeg", ".mpg", ".mpga", ".m4a", ".wav", ".webm", ".mkv", ".avi", ".flv", ".mov", ".wmv", ".3gp", ".3g2", ".vob", ".flac", ".aac", ".ogg", ".wma", ".ac3", ".dts", ".mmf", ".m4r", ".mp2", ".wv", ".asf", ".f4v", ".m2ts", ".mts", ".rm", ".rmvb", ".swf", ".wtv"]

# Available models for Whisper
AVAILABLE_MODELS: List[str] = ['tiny', 'base', 'small', 'medium', 'large-v2', 'large-v3']
AVAILABLE_LANGUAGES: List[str] = ["Afrikaans", "Arabic", "Armenian", "Azerbaijani", "Belarusian", "Bosnian", "Bulgarian", "Catalan", "Chinese", "Croatian", "Czech", "Danish", "Dutch", "English", "Estonian", "Finnish", "French", "Galician", "German", "Greek", "Hebrew", "Hindi", "Hungarian", "Icelandic", "Indonesian", "Italian", "Japanese", "Kannada", "Kazakh", "Korean", "Latvian", "Lithuanian", "Macedonian", "Malay", "Marathi", "Maori", "Nepali", "Norwegian", "Persian", "Polish", "Portuguese", "Romanian", "Russian", "Serbian", "Slovak", "Slovenian", "Spanish", "Swahili", "Swedish", "Tagalog", "Tamil", "Thai", "Turkish", "Ukrainian", "Urdu", "Vietnamese", "Welsh"]

# File extensions not directly supported by Whisper, requiring conversion to mp3
EXTENSIONS_REQUIRING_CONVERSION: List[str] = [".mkv", ".avi", ".flv", ".mov", ".wmv", ".3gp", ".3g2", ".vob", ".flac", ".aac", ".ogg", ".wma", ".ac3", ".dts", ".mmf", ".m4r", ".mp2", ".wv", ".asf", ".f4v", ".m2ts", ".mts", ".rm", ".rmvb", ".swf", ".wtv"]


def get_device() -> str:
    """Checks if a CUDA-compatible GPU is available and returns the device string.

    Returns:
        str: "cuda" if a CUDA-compatible GPU is available, otherwise "cpu".
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


def user_device_choice() -> str:
    """Prompts the user to choose between GPU and CPU for transcription.

    Returns:
        str: "Y" if the user chooses GPU, "N" if the user chooses CPU.
    """
    answ = input("\nDo you prefer to use the GPU for transcription?\nAnswer with 'Y' or 'N':  ").strip().upper()
    while answ != "Y" and answ != "N":
        answ = input("Answer correctly using 'Y' or 'N':  ").strip().upper()
    return answ


def get_model_name() -> str:
    """Prompts the user to choose a Whisper transcription model.

    Returns:
        str: The name of the chosen Whisper model (e.g., 'tiny', 'base').
    """
    answ = input(f"\nWhich model do you prefer to use?\nEnter one of the following models: {', '.join(AVAILABLE_MODELS)}.\nPlease note: refer to the whisper GitHub repository for information on performance and requirements.\n").strip().lower()
    while answ not in AVAILABLE_MODELS:
        answ = input("Please, answer correctly indicating one of the following models: 'tiny', 'base', 'small', 'medium', 'large-v2', 'large-v3'.\n").strip().lower()
    return answ


def get_input_path() -> str:
    """Prompts the user for the path to the input file or folder.

    The path must correspond to an existing file or directory.

    Returns:
        str: The validated path to the input file or folder.
    """
    answ = input("\nWrite the path to the file to be transcribed or to the folder containing all the files to be transcribed.\nNOTE: if it is a folder containing several files, take care that they do not all have the same name, otherwise the text file that will contain the transcription will always have the exact same name and will therefore be overwritten each time.\n").strip()
    while not os.path.exists(answ) or not (os.path.isfile(answ) or os.path.isdir(answ)): # Ensure it's a file or directory
        answ = input("The path entered does not correspond to any file or folder, please enter an existing path.\n").strip()
    return answ


def get_language() -> str:
    """Prompts the user to specify the language for transcription.

    The language must be one of the languages supported by Whisper.

    Returns:
        str: The capitalized name of the chosen language (e.g., "Italian").
    """
    answ = input(f"\nIndicate the language to be used for transcriptions.\nThe supported languages are:\n{AVAILABLE_LANGUAGES}\n").strip().capitalize()
    while answ not in AVAILABLE_LANGUAGES:
        answ = input("Please, answer correctly indicating one of the supported languages.\n").strip().capitalize()
    return answ


def get_output_path(single_file: bool) -> str:
    """Prompts the user for the path to the output folder for transcriptions.

    The path must correspond to an existing directory.

    Args:
        single_file (bool): True if a single file is being transcribed, False otherwise.
                            This affects the prompt message.

    Returns:
        str: The validated path to the output folder.
    """
    if single_file:
        prompt_message = "\nWrite the path to the folder where you want to save the transcription.\n"
    else:
        prompt_message = "\nWrite the path to the folder where you want to save the transcripts.\n"
    answ = input(prompt_message).strip()
    while not os.path.isdir(answ): # Check if it's an existing directory
        answ = input("Please, write an existing folder path.\n").strip()
    return answ


def get_order_criterion() -> str:
    """Prompts the user to choose an order criterion for transcribing multiple files.

    Returns:
        str: A string representing the chosen order criterion ("1" through "6").
    """
    available_choises = ["1", "2", "3", "4", "5", "6"]
    answ = input("\nYou can choose one of the following order criteria:\n1 - Transcribe files according to their creation date (oldest to newest);\n2 - Transcribe files according to their creation date (newest to oldest);\n3 - Transcribe files according to their last modification date (oldest to newest);\n4 - Transcribe files according to their last modification date (newest to oldest);\n5 - Customized order: enter numbers in the file names within round brackets (e.g. ‘file_name_(1).extension’) and the script will transcribe the files based on these, in ascending order.\n6 - Any order (doesn't matter).\nAnswer by writing only the number corresponding to the chosen order:  ").strip()
    while answ not in available_choises:
        answ = input("Please, answer correctly indicating the number of the chosen criterion: ").strip().lower()
    return answ


def extract_number(filename: str) -> Optional[int]:
    """Extracts a number enclosed in parentheses from a filename.

    Args:
        filename (str): The filename to parse.

    Returns:
        Optional[int]: The extracted number as an integer if found, otherwise None.
    """
    # Pattern to find numbers in round brackets
    pattern = r'\((\d+)\)'
    # Search for the pattern in the string
    match = re.search(pattern, filename)
    if match:
        # Return the number as an integer
        return int(match.group(1))
    # Return None if the pattern is not found
    return None


def convert_to_mp3(input_path: str, output_path: str) -> None:
    """Converts a media file to MP3 format using FFmpeg.

    This function calls the FFmpeg command-line tool to perform the conversion.
    Audio is extracted and saved with high quality.

    Args:
        input_path (str): Path to the input media file.
        output_path (str): Path where the output MP3 file will be saved.
    """
    try:
        # Determine the appropriate null device for the OS
        # For Windows
        if os.name == 'nt':
            null_device = 'nul'
        # For Unix/Linux/MacOS
        else:
            null_device = '/dev/null'
        # Construct the ffmpeg command
        command = ['ffmpeg', '-i', input_path, '-q:a', '0', '-map', 'a', output_path]
        # Execute the command and suppress output generated by ffmpeg
        # Adding check=True to raise CalledProcessError on failure
        with open(null_device, 'w') as devnull:
            subprocess.run(command, stderr=devnull, check=True)
        print(f"\n{datetime.now().strftime('%d/%m/%Y %H:%M')} - Conversion completed: {input_path} -> {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"\n{datetime.now().strftime('%d/%m/%Y %H:%M')} - Error converting file {input_path} to mp3: {e}")


def transcribe_single_file(
    model: WhisperModel,
    file: str,
    in_path: str,
    out_path: str,
    language: str
) -> None:
    """Transcribes a single audio/video file using the provided Whisper model.

    If the file format is not directly supported by Whisper, it attempts to convert
    it to MP3 first. The transcription is saved as a .txt file in the output path.

    Args:
        model (WhisperModel): The loaded Whisper model instance.
        file (str): The name of the file to transcribe (e.g., "audio.wav").
        in_path (str): The directory path where the input file is located.
        out_path (str): The directory path where the transcription text file will be saved.
        language (str): The language of the audio content for transcription.
    """
    # This allows to immediately understand, from the console, which file the program is working on
    print(f"\n{file}:")

    # Storage in variables of the path to the file to be transcribed, its extension and the path to which the text file containing the transcription will be saved
    file_path = os.path.join(in_path, file)
    file_extension = os.path.splitext(file)[1]
    txt_path = os.path.join(out_path, os.path.splitext(file)[0] + ".txt")
    
    if file_extension.lower() in EXTENSIONS_SUPPORTED:
            
        if file_extension.lower() in EXTENSIONS_REQUIRING_CONVERSION:
            print(f"\n{datetime.now().strftime('%d/%m/%Y %H:%M')} - Starting conversion of file {file} from '{file_extension}' to '.mp3'...")
            mp3_path = os.path.join(in_path, os.path.splitext(file)[0] + ".mp3")
            # Only if the mp3 version of the file does not already exist, then it will be converted
            if not os.path.exists(mp3_path):
                convert_to_mp3(file_path, mp3_path)
            else: print(f"\n{datetime.now().strftime('%d/%m/%Y %H:%M')} - There is already an 'mp3' version of the file {file}, so the conversion was not made.")
            file_path = mp3_path
        
        print(f"\n{datetime.now().strftime('%d/%m/%Y %H:%M')} - Starting transcription of file {file}...")

        try:
            # Transcribe the audio
            # The line "file_path = os.path.join(in_path, file)" was removed from here
            result = model.transcribe(file_path, language=language) # Uses the potentially converted file_path

            # Save the transcription to a text file
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(result["text"])
            
            print(f"\n{datetime.now().strftime('%d/%m/%Y %H:%M')} - Transcription of file {file} completed successfully!\n\n------------------------------------------------------")
        except Exception as e:
            print(f"\n{datetime.now().strftime('%d/%m/%Y %H:%M')} - Error transcribing file {file}: {e}\n\n------------------------------------------------------")
    else:
        print(f"\n{datetime.now().strftime('%d/%m/%Y %H:%M')} - Skipping unsupported file type: {file}")


def transcribe_multiple_files(
    model: WhisperModel,
    in_path: str,
    out_path: str,
    language: str,
    order_criterion: str
) -> None:
    """Transcribes all supported files in a given directory.

    Files are filtered by supported extensions and then ordered according
    to the user-specified criterion before being transcribed one by one.

    Args:
        model (WhisperModel): The loaded Whisper model instance.
        in_path (str): The directory path containing the files to transcribe.
        out_path (str): The directory path where transcription text files will be saved.
        language (str): The language of the audio content for transcription.
        order_criterion (str): The criterion for ordering files before transcription
                               (e.g., "1" for creation date, "5" for custom numbering).
    """
    # Consider all files in the folder specified by the user as the input folder
    files = os.listdir(in_path)
    # We filter the list of files to be transcribed, keeping only those that are actually files (excluding directories) and whose extension falls within the supported extensions defined at the top of the code
    files = [f for f in files if os.path.isfile(os.path.join(in_path, f)) and os.path.splitext(f)[1].lower() in EXTENSIONS_SUPPORTED]

    # Reading the order criterion specified by the user and using it
    if order_criterion == "1":
        files.sort(key=lambda x: os.path.getctime(os.path.join(in_path, x)))
    elif order_criterion == "2":
        files.sort(key=lambda x: os.path.getctime(os.path.join(in_path, x)), reverse=True)
    elif order_criterion == "3":
        files.sort(key=lambda x: os.path.getmtime(os.path.join(in_path, x)))
    elif order_criterion == "4":
        files.sort(key=lambda x: os.path.getmtime(os.path.join(in_path, x)), reverse=True)
    elif order_criterion == "5":
        # In the case of customised orders, it may happen that some files do not have the order number written inside their name, in which case they are converted last
        files.sort(key=lambda x: extract_number(x) if extract_number(x) is not None else float('inf'))

    for file in files:
        transcribe_single_file(model, file, in_path, out_path, language)


def main() -> None:
    """Main function to run the Whisper Queue transcription program.

    Guides the user through setup (device, model, paths, language, order)
    and then starts the transcription process for a single file or multiple files.
    """
    # Checking that a CUDA-compatible GPU is available
    device = get_device()

    # Ask the user whether they prefer to use the GPU or the CPU
    answ = user_device_choice()
    if answ == "N":
        device = "cpu"

    # Ask the user to choose the Whisper model to use for the transcription
    model_name = get_model_name()
    
    # Load the Whisper model once
    print(f"\n{datetime.now().strftime('%d/%m/%Y %H:%M')} - Loading Whisper model '{model_name}' on device '{device}'...")
    model = whisper.load_model(model_name, device=device)
    print(f"\n{datetime.now().strftime('%d/%m/%Y %H:%M')} - Model loaded successfully.")

    # Ask the user to specify the path to the file/folder to be transcribed
    in_path = get_input_path()

    # Ask the user to specify the language to use for the transcription
    language = get_language()

    # Check if the input path is a single file or a folder
    if os.path.isfile(in_path):
        single_file = True
    else:
        single_file = False
    
    # Ask the user to specify the path to the output folder
    out_path = get_output_path(single_file)

    # We use the appropriate function depending on whether we need to transcribe a single file or multiple files
    if single_file:
        file = os.path.basename(in_path)
        in_path = os.path.dirname(in_path)
        print("\n------------------------------------------------------")
        transcribe_single_file(model, file, in_path, out_path, language)
    else:
        # In the case of multiple files, we ask the user to specify the order criterion
        order_criterion = get_order_criterion()
        print("\n------------------------------------------------------")
        transcribe_multiple_files(model, in_path, out_path, language, order_criterion)

if __name__ == "__main__":
    main()