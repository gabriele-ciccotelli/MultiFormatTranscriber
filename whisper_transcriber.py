from datetime import datetime
from moviepy.editor import *
import subprocess
import whisper
import torch
import re
import os


# Supported extensions
extensions = [".mp3", ".mp4", ".mpeg", ".mpg", ".mpga", ".m4a", ".wav", ".webm", ".mkv", ".avi", ".flv", ".mov", ".wmv", ".3gp", ".3g2", ".vob", ".flac", ".aac", ".ogg", ".wma", ".ac3", ".dts", ".mmf", ".m4r", ".mp2", ".wv", ".asf", ".f4v", ".m2ts", ".mts", ".rm", ".rmvb", ".swf", ".wtv"]

# Function to check if the machine has a CUDA compatible GPU
def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

# Function to allow the user to choose between GPU and CPU
def user_device_choice():
    answ = input("\nDo you prefer to use the GPU for transcription?\nAnswer with 'Y' or 'N':  ").strip().upper()
    while answ != "Y" and answ != "N":
        answ = input("Answer correctly using 'Y' or 'N':  ").strip().upper()
    return answ

# Function to allow the user to choose which transcription model to use
def get_model_name():
    available_models = ['tiny', 'base', 'small', 'medium', 'large-v2', 'large-v3']
    answ = input("\nWhich model do you prefer to use?\nEnter one of the following models: 'tiny', 'base', 'small', 'medium', 'large-v2', 'large-v3'.\nPlease note: refer to the whisper GitHub repository for information on performance and requirements.\n").strip().lower()
    while answ not in available_models:
        answ = input("Please, answer correctly indicating one of the following models: 'tiny', 'base', 'small', 'medium', 'large-v2', 'large-v3'.\n").strip().lower()
    return answ

# Function to obtain the path to the folder containing the files to be transcribed or to the individual file to be transcribed
def get_input_path():
    answ = input("\nWrite the path to the file to be transcribed or to the folder containing all the files to be transcribed.\nNOTE: if it is a folder containing several files, take care that they do not all have the same name, otherwise the text file that will contain the transcription will always have the exact same name and will therefore be overwritten each time.\n").strip()
    while not os.path.exists(answ):
        answ = input("The path entered does not correspond to any file or folder, please enter an existing path.\n").strip()
    return answ

# Function to obtain the language to be used for transcription from the user
def get_language():
    available_languages = ["Afrikaans", "Arabic", "Armenian", "Azerbaijani", "Belarusian", "Bosnian", "Bulgarian", "Catalan", "Chinese", "Croatian", "Czech", "Danish", "Dutch", "English", "Estonian", "Finnish", "French", "Galician", "German", "Greek", "Hebrew", "Hindi", "Hungarian", "Icelandic", "Indonesian", "Italian", "Japanese", "Kannada", "Kazakh", "Korean", "Latvian", "Lithuanian", "Macedonian", "Malay", "Marathi", "Maori", "Nepali", "Norwegian", "Persian", "Polish", "Portuguese", "Romanian", "Russian", "Serbian", "Slovak", "Slovenian", "Spanish", "Swahili", "Swedish", "Tagalog", "Tamil", "Thai", "Turkish", "Ukrainian", "Urdu", "Vietnamese", "Welsh"]
    answ = input(f"\nIndicate the language to be used for transcriptions.\nThe supported languages are:\n{available_languages}\n").strip().capitalize()
    while answ not in available_languages:
        answ = input("Please, answer correctly indicating one of the supported languages.\n").strip().capitalize()
    return answ

# Function to obtain the path to the folder where the text files with the transcripts will be saved
def get_output_path(single_file):
    if single_file:
        answ = input("\nWrite the path to the folder where you want to save the transcription.\n").strip()
        while not os.path.exists(answ):
            answ = input("Please, write an existing path.\n").strip()
    else:
        answ = input("\nWrite the path to the folder where you want to save the transcripts.\n").strip()
        while not os.path.exists(answ):
            answ = input("Please, write an existing path.\n").strip()
    return answ

# Function to get the order criterion from the user
def get_order_criterion():
    available_choises = ["1", "2", "3", "4", "5", "6"]
    answ = input("\nYou can choose one of the following order criteria:\n1 - Transcribe files according to their creation date (oldest to newest);\n2 - Transcribe files according to their creation date (newest to oldest);\n3 - Transcribe files according to their last modification date (oldest to newest);\n4 - Transcribe files according to their last modification date (newest to oldest);\n5 - Customized order: enter numbers in the file names within round brackets (e.g. ‘file_name_(1).extension’) and the script will transcribe the files based on these, in ascending order.\n6 - Any order (doesn't matter).\nAnswer by writing only the number corresponding to the chosen order:  ").strip()
    while answ not in available_choises:
        answ = input("Please, answer correctly indicating the number of the chosen criterion:  ").strip().lower()
    return answ

# Function to extract numbers from filenames
def extract_number(string):
    # Pattern to find numbers in round brackets
    pattern = r'\((\d+)\)'
    # Search for the pattern in the string
    match = re.search(pattern, string)
    if match:
        # Return the number as an integer
        return int(match.group(1))
    # Return None if the pattern is not found
    return None

# Function to convert audio to the mp3 format, that is supported by Whisper
def convert_to_mp3(input_path, output_path):
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
        with open(null_device, 'w') as devnull:
            subprocess.run(command, stderr=devnull)
        print(f"\n{datetime.now().strftime('%d/%m/%Y %H:%M')} - Conversion completed: {input_path} -> {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"\n{datetime.now().strftime('%d/%m/%Y %H:%M')} - Error converting file {input_path} to mp3: {e}")

# Function to transcribe a single file
def transcribe_single_file(model_name, device, file, in_path, out_path, language):
    # This allows to immediately understand, from the console, which file the program is working on
    print(f"\n{file}:")
    # This instruction allows to load the model chosen by the user
    model = whisper.load_model(model_name, device=device)

    # Storage in variables of the path to the file to be transcribed, its extension and the path to which the text file containing the transcription will be saved
    file_path = os.path.join(in_path, file)
    file_extension = os.path.splitext(file)[1]
    txt_path = os.path.join(out_path, os.path.splitext(file)[0] + ".txt")
    
    # These are the file extensions not supported by Whisper for transcription, so if the file has one of these extensions, the program will convert it in 'mp3' format
    unsupported_extensions = [".mkv", ".avi", ".flv", ".mov", ".wmv", ".3gp", ".3g2", ".vob", ".flac", ".aac", ".ogg", ".wma", ".ac3", ".dts", ".mmf", ".m4r", ".mp2", ".wv", ".asf", ".f4v", ".m2ts", ".mts", ".rm", ".rmvb", ".swf", ".wtv"]

    if file_extension.lower() in extensions:
            
        if file_extension.lower() in unsupported_extensions:
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
            file_path = os.path.join(in_path, file)
            result = model.transcribe(file_path, language=language)

            # Save the transcription to a text file
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(result["text"])
            
            print(f"\n{datetime.now().strftime('%d/%m/%Y %H:%M')} - Transcription of file {file} completed successfully!\n\n------------------------------------------------------")
        except Exception as e:
            print(f"\n{datetime.now().strftime('%d/%m/%Y %H:%M')} - Error transcribing file {file}: {e}\n\n------------------------------------------------------")
    else:
        print(f"\n{datetime.now().strftime('%d/%m/%Y %H:%M')} - Skipping unsupported file type: {file}")

# Function to transcribe multiple files
def transcribe_multiple_files(model_name, device, in_path, out_path, language, order_criterion):
    
    # Consider all files in the folder specified by the user as the input folder
    files = os.listdir(in_path)
    # We filter the list of files to be transcribed, keeping only those that are actually files (excluding directories) and whose extension falls within the supported extensions defined at the top of the code
    files = [file for file in files if os.path.isfile(os.path.join(in_path, file)) and os.path.splitext(file)[1].lower() in extensions]

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
        transcribe_single_file(model_name, device, file, in_path, out_path, language)

# Main script
def main():
    # Checking that a CUDA-compatible GPU is available
    device = get_device()

    # Ask the user whether they prefer to use the GPU or the CPU
    answ = user_device_choice()
    if answ == "N":
        device = "cpu"

    # Ask the user to choose the Whisper model to use for the transcription
    model_name = get_model_name()

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
        transcribe_single_file(model_name, device, file, in_path, out_path, language)
    else:
        # In the case of multiple files, we ask the user to specify the order criterion
        order_criterion = get_order_criterion()
        print("\n------------------------------------------------------")
        transcribe_multiple_files(model_name, device, in_path, out_path, language, order_criterion)

if __name__ == "__main__":
    main()