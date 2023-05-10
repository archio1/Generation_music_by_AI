import os
import logging
import numpy as np
import json
from pathlib import Path
from typing import List, Dict
from music21 import note, chord, instrument, converter, stream
from sklearn.model_selection import train_test_split
from create_train_model import MusicModel
from generate_notes import NotesGenerator
from read_and_convert_file import MusicProcessor
import keras.utils as np_util

from keras.models import Sequential, load_model
from keras.layers import LSTM, Dropout, Dense

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
    datefmt='%m-%d %H:%M',
    filename='data_processing.log',
    filemode='w',
)


#
# def create_input_output_arrays(notes_per_file, notes_with_indexes, timesteps=50):
#     x = []
#     y = []
#     for notes in notes_per_file:
#         for j in range(0, len(notes) - timesteps):
#             inp = notes[j : j + timesteps]
#             out = notes[j + timesteps]
#             x.append(list(map(lambda x: notes_with_indexes[x], inp)))
#             y.append(notes_with_indexes[out])
#     x_new = np.array(x)
#     y_new = np.array(y)
#     x_new = np.reshape(x_new, (len(x_new), timesteps, 1))
#     y_new = np.reshape(y_new, (-1, 1))
#
#     return x_new, y_new


# def create_input_output_sequences(notes: List[str], sequence_length: int, pitch_to_int: Dict[str, int]) -> Tuple[np.ndarray, np.ndarray]:
#     pitch_names = sorted(set(notes))
#     pitch_to_int = {pitch: num for num, pitch in enumerate(pitch_names)}
#     input_sequences = []
#     output_sequences = []
#
#     for i in range(0, len(notes) - sequence_length, 1):
#         sequence_in = notes[i:i + sequence_length]
#         sequence_out = notes[i + sequence_length]
#         input_sequences.append([pitch_to_int[char] for char in sequence_in])
#         output_sequences.append(pitch_to_int[sequence_out])
#
#     n_patterns = len(input_sequences)
#
#     input_sequences = np.reshape(input_sequences, (n_patterns, sequence_length, 1))
#     input_sequences = input_sequences / float(len(pitch_names))
#
#     output_sequences = np_utils.to_categorical(output_sequences)
#
#     return input_sequences, output_sequences
#
# def get_train_test_sets(x_new, y_new, timesteps=50):
#     x_train, x_test, y_train, y_test = train_test_split(
#         x_new, y_new, test_size=0.2, random_state=42
#     )
#
#     return x_train, x_test, y_train, y_test

# def prepare_sequences(notes, sequence_length):
#
#     pitch_names = sorted(set(item for item in notes))
#
#     pitch_to_int = dict((pitch, number) for number, pitch in enumerate(pitch_names))
#     with open('../data/pitch_to_int.json', 'w') as f:
#         json.dump(pitch_to_int, f, indent=4)
#
#     network_input = []
#     network_output = []
#
#     for i in range(0, len(notes) - sequence_length):
#         sequence_in = notes[i:i + sequence_length]
#         sequence_out = notes[i + sequence_length]
#         network_input.append([pitch_to_int[char] for char in sequence_in])
#         network_output.append(pitch_to_int[sequence_out])
#
#     n_patterns = len(network_input)
#
#     normalized_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
#
#     normalized_input = normalized_input / float(len(pitch_names))
#
#     # split the data into training and test sets
#     x_train, x_test, y_train, y_test = train_test_split(normalized_input, network_output, test_size=0.2,
#                                                         random_state=42)
#
#     return x_train, x_test, y_train, y_test, normalized_input, pitch_names

def prepare_sequences(notes, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    sequence_length = 100

    # get all pitch names
    pitchnames = sorted(set(item for item in notes))

    # create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    network_output = np.reshape(network_output, (-1, 1))
    # normalize input
    # network_input = network_input / float(n_vocab)

    # network_output = np_util.to_categorical(network_output)

    return network_input, network_output


def get_train_test_sets(network_input, network_output):
    x_train, x_test, y_train, y_test = train_test_split(
        network_input, network_output, test_size=0.2, random_state=42
    )

    return x_train, x_test, y_train, y_test

if __name__ == '__main__':
    notes = []
    all_notes = []
    notes_per_file = []

    # Reading and extrating data
    music_processor = MusicProcessor(r'../data/midi/bach')
    midi_files = music_processor.get_midi_files()
    audio_streams = [music_processor.convert_midi_to_audio_stream(midi_file) for midi_file in midi_files]
    for audio_stream in audio_streams:
        notes = music_processor.extract_piano_notes(audio_stream)
        all_notes.extend(notes)
        notes_per_file.append(notes)

    # Analysis and selection of data
    # unique_notes = get_unique_notes(all_notes)
    # notes_frequency = get_notes_frequency(unique_notes, all_notes)
    # notes_frequency = get_notes_frequency_filtered_by_threshold(notes_frequency)
    # filtered_notes_per_file = get_filtered_notes_by_frequency(
    #     notes_frequency, notes_per_file
    # )
    # indexes_with_notes = get_indexes_with_notes(notes_frequency)
    # notes_with_indexes = get_notes_with_indexes(indexes_with_notes)

    # Data pre-processing
    n_vocab = len(set(all_notes))
    pitchnames = sorted(set(item for item in all_notes))
    network_input, network_output = prepare_sequences(all_notes, n_vocab)
    x_train, x_test, y_train, y_test = get_train_test_sets(network_input, network_output)
    # Model
    create_model = MusicModel(n_vocab)
    create_model.create_and_train_model(network_input)
    x_train_list = x_train.tolist()
    x_test_list = x_test.tolist()
    create_model.train_model(x_train, x_test, y_train, y_test)
    create_model.save_model()
    gen_music = NotesGenerator('s2s')
    gen_notes = gen_music.generate_notes(network_input, pitchnames, n_vocab)
    gen_music.create_midi(gen_notes)
