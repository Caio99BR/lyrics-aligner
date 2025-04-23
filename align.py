import argparse
import os
import pickle
import librosa as lb
import torch
import numpy as np
from praatio import textgrid

import model

def compute_phoneme_onsets(opt_path_matrix, hop_len, samp_rate):
    """
    Compute phoneme onset times based on the optimal alignment matrix.
    """
    phoneme_idx = np.argmax(opt_path_matrix, axis=1)

    # Indices where a phoneme jump occurred (space between two consecutive phonemes)
    skipped_idx = [x + 1 for i, (x, y) in enumerate(zip(phoneme_idx[:-1], phoneme_idx[1:])) if x == y - 2]

    # Indices where phoneme change occurs
    last_idx_change = [i for i, (x, y) in enumerate(zip(phoneme_idx[:-1], phoneme_idx[1:])) if x != y]

    phoneme_onsets = [(n + 1) * hop_len / samp_rate for n in last_idx_change]
    phoneme_onsets.insert(0, 0)  # First phoneme starts at 0s

    if skipped_idx:
        for idx in skipped_idx:
            phoneme_onsets.insert(idx, phoneme_onsets[idx] + (hop_len / samp_rate))

    return phoneme_onsets

def compute_word_alignment(phonemes, phoneme_onsets):
    """
    Align phonemes to words and return word onset and offset times.
    """
    word_onsets, word_offsets = [], []

    for idx, phoneme in enumerate(phonemes):
        if idx == 0:
            word_onsets.append(phoneme_onsets[1])
            continue
        if phoneme == '>' and idx != len(phonemes) - 1:
            word_offsets.append(phoneme_onsets[idx])
            word_onsets.append(phoneme_onsets[idx + 1])

    word_offsets.append(phoneme_onsets[-1])
    return word_onsets, word_offsets

def accumulated_cost_numpy(score_matrix, init=None):
    """
    Compute the accumulated cost matrix using DTW.
    """
    B, N, M = score_matrix.size()
    score_matrix = score_matrix.numpy().astype('float64')

    dtw_matrix = np.ones((N + 1, M + 1)) * -1e5
    dtw_matrix[0, 0] = init

    for (m, n), (m_m1, n_m1) in zip(model.MatrixDiagonalIndexIterator(m=M + 1, n=N + 1, k_start=1),
                                    model.MatrixDiagonalIndexIterator(m=M, n=N, k_start=0)):
        max_values = np.maximum(dtw_matrix[n_m1, m], dtw_matrix[n_m1, m_m1])
        dtw_matrix[n, m] = score_matrix[0, n_m1, m_m1] + max_values

    return dtw_matrix[1:N + 1, 1:M + 1]

def optimal_alignment_path(matrix, init=200):
    """
    Perform DTW and return the binary optimal path matrix.
    """
    accumulated_scores = accumulated_cost_numpy(matrix, init=init)
    N, M = accumulated_scores.shape

    path = np.zeros((N, M))
    path[-1, -1] = 1

    n, m = N - 2, M - 1
    while m > 0:
        d1 = accumulated_scores[n, m]
        d2 = accumulated_scores[n, m - 1]
        arg_max = np.argmax([d1, d2])
        path[n, m - arg_max] = 1
        n -= 1
        m -= arg_max
        if n == -2:
            print(f"DTW failed. n={n}, m={m}")
            break
    path[0:n+1, 0] = 1
    return path

def make_phoneme_and_word_list(text_file, word2phoneme_dict):
    """
    Extract word and phoneme symbol lists from a word-based lyrics file.
    """
    word_list, phonemes = [], ['>']
    with open(text_file, encoding='utf-8') as f:
        for line in f:
            line = line.lower().replace('\n', '').replace('’', "'")
            clean_line = ''.join(c for c in line if c.isalnum() or c in ["'", ' '])
            if not clean_line.strip():
                continue
            for word in clean_line.split():
                word_list.append(word)
                phonemes += word2phoneme_dict[word].split(' ') + ['>']
    return phonemes, word_list

def make_phoneme_list(text_file):
    """
    Extract phoneme list directly from a phoneme-based lyrics file.
    """
    with open(text_file, encoding='utf-8') as f:
        return [line.strip().upper() for line in f if line.strip() and line.strip() != ' ']

def convert_onsets_tsv_to_textgrid(onset_file_path, grid_file_path):
    """
    Converts phoneme onsets TSV file to Praat TextGrid format.
    """
    pointList = []
    with open(onset_file_path, 'r') as fi:
        for line in fi:
            phoneme, onset = line.split('\t')
            phoneme = '' if phoneme == '>' else phoneme
            pointList.append((phoneme, float(onset)))

    # Add the last point for duration
    duration = pointList[-1][1] + 1.0
    pointList.append(('', duration))

    # Create intervals
    newEntries = [(pointList[i][1], pointList[i + 1][1], pointList[i][0]) for i in range(len(pointList) - 1)]

    # Create TextGrid
    outputTG = textgrid.Textgrid()
    tier = textgrid.IntervalTier("phons", newEntries, 0, duration)
    outputTG.addTier(tier)

    # Save to file
    outputTG.save(grid_file_path, "short_textgrid", True)

def find_files_in_directory(input_dir):
    """
    Recursively finds all text and audio files in the given directory and its subdirectories.
    """
    lyrics_files, audio_files = [], []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.txt'):
                lyrics_files.append(os.path.join(root, file))
            elif file.endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a')):
                audio_files.append(os.path.join(root, file))
    return lyrics_files, audio_files


def process_onset_files(onset_type, output_file_dir, file_name, phonemes, phoneme_onsets, dataset_name, words=None):
    """
    Process onset files (phoneme or word onsets) and save them in the appropriate directory.
    """
    onset_dirs = {'p': 'phoneme_onsets', 'w': 'word_onsets'}
    onset_dir = os.path.join(output_file_dir, f"{dataset_name}_{onset_dirs[onset_type]}")

    # Create the directory only once if it doesn't exist
    os.makedirs(onset_dir, exist_ok=True)

    # Write onset times to the corresponding file
    onset_file_path = os.path.join(onset_dir, f'{file_name}.txt')
    with open(onset_file_path, 'w') as f:
        for item, onset in zip(phonemes if onset_type == 'p' else words, phoneme_onsets):
            f.write(f'{item}\t{onset}\n')

    # Convert phoneme onsets to TextGrid if phonemes are being processed
    if onset_type == 'p':
        grid_file_path = os.path.join(onset_dir, f'{file_name}_TextGrid.txt')
        convert_onsets_tsv_to_textgrid(onset_file_path, grid_file_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Lyrics aligner by phoneme/word')
    parser.add_argument('input_dir', type=str, default='inputs', nargs='?', help='Directory containing both audio and lyrics')
    parser.add_argument('--lyrics-format', choices=['w', 'p'], default='w')
    parser.add_argument('--onsets', choices=['p', 'w', 'pw'], default='p')
    parser.add_argument('--dataset-name', default='dataset1')
    parser.add_argument('--vad-threshold', type=float, default=0)
    args = parser.parse_args()

    # Find lyrics and audio files recursively within the input directory
    lyrics_files, audio_files = find_files_in_directory(args.input_dir)

    # Load necessary files for phoneme to index and word to phoneme dictionary
    with open(f'bin/{args.dataset_name}_word2phonemes.pickle', 'rb') as f:
        word2phonemes = pickle.load(f)
    with open('bin/phoneme2idx.pickle', 'rb') as f:
        phoneme2idx = pickle.load(f)

    # Load the pre-trained model
    state_dict = torch.load('bin/model_parameters.pth', map_location='cpu', weights_only=True)
    model_instance = model.InformedOpenUnmix3()
    model_instance.load_state_dict(state_dict)

    # Create output directory structure
    output_dir = f'outputs/'

    # Process each lyrics file and find the corresponding audio
    for lyrics_path in lyrics_files:
        file_name = os.path.splitext(os.path.basename(lyrics_path))[0]

        # Try to find the corresponding audio file
        audio_path = next((audio for audio in audio_files if os.path.splitext(os.path.basename(audio))[0] == file_name), None)

        if audio_path is None:
            print(f"Error: Audio file for {lyrics_path} not found. Skipping.")
            continue  # Skip this iteration and go to the next one

        if len([audio for audio in audio_files if os.path.splitext(os.path.basename(audio))[0] == file_name]) > 1:
            print(f"Error: Multiple audio files found for {lyrics_path}. Skipping.")
            continue  # Skip this iteration and go to the next one

        # Create a folder for the output
        output_file_dir = os.path.join(output_dir, file_name)

        # Check and print audio format preference
        print(f"Processing: {audio_path} <+> {lyrics_path} = {output_file_dir}")
        if not audio_path.endswith('.wav'):
            print(f"  Warning: Use Audio File on .wav format, if possible (currently using .{audio_path.split('.')[-1].upper()})")

        # Process lyrics based on the selected format
        if args.lyrics_format == 'w':
            print(f'  Using dictionary: bin/{args.dataset_name}_word2phonemes.pickle')
            phonemes, words = make_phoneme_and_word_list(lyrics_path, word2phonemes)
        else:
            phonemes = make_phoneme_list(lyrics_path)

        # Convert phonemes to indices and prepare for model input
        phoneme_idx = [phoneme2idx[p] for p in phonemes]
        phoneme_tensor = torch.tensor(phoneme_idx, dtype=torch.float32).unsqueeze(0)

        # Load and resample the audio file
        audio, sr = lb.load(audio_path, sr=None, mono=True)
        if sr != 16000:
            audio = lb.resample(audio, orig_sr=sr, target_sr=16000)

        audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            voice_estimate, _, scores = model_instance((audio_tensor, phoneme_tensor))
            scores = scores.cpu()

        # Perform Voice Activity Detection (VAD) if necessary
        if args.vad_threshold > 0:
            voice_mag = voice_estimate[:, 0, 0, :].cpu().numpy().T
            silence_frames = np.where(np.sum(voice_mag, axis=0) < args.vad_threshold)[0]
            space_token_idx = torch.nonzero(phoneme_tensor == 3, as_tuple=True)[1]
            for n in silence_frames:
                scores[:, n, space_token_idx] = scores.max()

        # Find the optimal alignment path for phonemes
        path = optimal_alignment_path(scores)

        # Compute phoneme onsets and process them
        phoneme_onsets = compute_phoneme_onsets(path, hop_len=256, samp_rate=16000)

        # Process and save phoneme and word onsets if requested
        if args.onsets == 'p' or args.onsets == 'pw':
            process_onset_files('p', output_file_dir, file_name, phonemes, phoneme_onsets, args.dataset_name)

        if args.onsets == 'w' or args.onsets == 'pw':
            word_onsets, word_offsets = compute_word_alignment(phonemes, phoneme_onsets)
            process_onset_files('w', output_file_dir, file_name, phonemes, word_onsets, args.dataset_name, words)

        print('Done.')
