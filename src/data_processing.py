import os
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict
from music21 import note, chord, instrument, converter, stream
from sklearn.model_selection import train_test_split
from create_train_model import MusicModel
from generate_notes import NotesGenerator
from read_and_convert_file import MusicProcessor
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dropout, Dense

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
    datefmt='%m-%d %H:%M',
    filename='data_processing.log',
    filemode='w',
)


# This function takes a folder path as input and reads all MIDI files in it
# It returns a list of audio streams for each MIDI file
# def read_midi_file(folder_path):
#     audio_streams = []
#     if os.path.exists(folder_path):
#         for dir_path, dir_names, filenames in os.walk(folder_path):
#             for filename in filenames:
#                 # check if the file ends with ".mid" or ".midi"
#                 if filename.endswith('.mid') or filename.endswith('.midi'):
#                     # read the MIDI file and append the audio stream to the list
#                     try:
#                         audio_stream = converter.parse(os.path.join(dir_path, filename))
#                         audio_streams.append(audio_stream)
#                     except KeyError:
#                         logging.debug(f'[KeyError] Fail of converting for {filename}')
#     else:
#         print(f'Current path not found: \n{os.path.abspath(folder_path)} ')
#         exit()
#     return audio_streams


# def get_midi_files(folder_path):
#     midi_files = []
#     folder_path = Path(folder_path)
#     if folder_path.exists():
#         for file_path in folder_path.rglob('**/*.mid'):
#             midi_files.append(file_path)
#     else:
#         print(f'Current path not found: \n{os.path.abspath(folder_path)} ')
#         exit()
#     return midi_files
#
#
# def convert_midi_to_audio_stream(midi_files):
#     audio_streams = []
#     for midi_file in midi_files:
#         try:
#             audio_stream = converter.parse(midi_file)
#             audio_streams.append(audio_stream)
#             return audio_stream
#         except KeyError:
#             logging.debug(f'[KeyError] Fail of converting for {midi_file}')
#             return None
#
#
# def extract_piano_notes(audio_stream: stream.Score) -> List[str]:
#     """Takes an audio stream and extracts piano notes from it.
#
#     Args:
#         audio_stream (stream.Score): container of music21 objects
#
#     Returns:
#         List[str]: converted list of notes
#     """
#     converted_notes = []
#     # get stream grouped by each unique instrument
#     audio_stream = instrument.partitionByInstrument(audio_stream)
#     for part in audio_stream.parts:
#         if 'Piano' in str(part):
#             # get elements (notes, chords, tempos, meters ...) from stream
#             part_elements = part.recurse()
#             for element in part_elements:
#                 if isinstance(element, note.Note):
#                     converted_notes.append(str(element.pitch))
#                 elif isinstance(element, chord.Chord):
#                     # append the pitches of the chord to the notes list
#                     converted_notes.append(
#                         '.'.join(str(notes) for notes in element.normalOrder)
#                     )
#     return converted_notes


def get_unique_notes(notes: List[str]) -> List[str]:
    """Get a list of unique notes (non-repeating notes)"""

    unique_notes = list(set(notes))
    print('Unique Notes:', len(unique_notes))

    return unique_notes


def get_notes_frequency(unique_notes: List[str], notes: List[str]) -> Dict[str, int]:
    """Get the frequency of each note"""

    notes_frequency = dict(map(lambda note: (note, notes.count(note)), unique_notes))

    return notes_frequency


def get_notes_frequency_filtered_by_threshold(
    notes_frequency: Dict[str, int], threshold: int = 50
) -> Dict[str, int]:
    """Get the number of notes whose frequency is above a certain threshold"""

    # check different threshold values (30, 50, 70, and 90)
    print("Notes frequency (threshold:number of notes):")
    for threshold_value in range(30, 100, 20):
        filtered_notes = [
            note for note in notes_frequency.values() if note >= threshold_value
        ]
        print(f"  {threshold_value}:{len(filtered_notes)}")

    notes_frequency_filtered_by_threshold = {
        note: frequency
        for note, frequency in notes_frequency.items()
        if frequency >= threshold
    }

    return notes_frequency_filtered_by_threshold


def get_filtered_notes_by_frequency(
    notes_frequency: Dict[str, int], notes_per_file: List[List[str]]
) -> List[str]:
    """Filter notes by frequency"""

    filtered_notes = [
        [note for note in notes if note in notes_frequency] for notes in notes_per_file
    ]

    return filtered_notes


def get_indexes_with_notes(notes_frequency: Dict[str, int]) -> Dict[int, str]:
    """Get a dictionary with keys as note indexes and values as notes themselves"""

    indexes_with_notes = dict(enumerate(notes_frequency))

    return indexes_with_notes


def get_notes_with_indexes(indexes_with_notes: Dict[int, str]) -> Dict[str, int]:
    """Get an inverted dictionary with keys as notes themselves and values as note indixes"""

    notes_with_indexes = dict(map(reversed, indexes_with_notes.items()))

    return notes_with_indexes


def create_input_output_arrays(notes_per_file, notes_with_indexes, timesteps=50):
    x = []
    y = []
    for notes in notes_per_file:
        for j in range(0, len(notes) - timesteps):
            inp = notes[j : j + timesteps]
            out = notes[j + timesteps]
            x.append(list(map(lambda x: notes_with_indexes[x], inp)))
            y.append(notes_with_indexes[out])
    x_new = np.array(x)
    y_new = np.array(y)
    x_new = np.reshape(x_new, (len(x_new), timesteps, 1))
    y_new = np.reshape(y_new, (-1, 1))

    return x_new, y_new


def get_train_test_sets(x_new, y_new, timesteps=50):
    x_train, x_test, y_train, y_test = train_test_split(
        x_new, y_new, test_size=0.2, random_state=42
    )

    return x_train, x_test, y_train, y_test


# def create_and_train_model(x_new, notes_with_indexes, x_train, x_test, y_train, y_test):
#     # Why is this type of model?
#     model = Sequential()
#     # 2 layer LSTM?
#     model.add(
#         LSTM(256, return_sequences=True, input_shape=(x_new.shape[1], x_new.shape[2]))
#     )
#     model.add(Dropout(0.2))
#     model.add(LSTM(256))
#     model.add(Dropout(0.2))
#     # This activation function?
#     model.add(Dense(256, activation='relu'))  # f(x) = max(0, x)
#     model.add(
#         Dense(len(notes_with_indexes), activation='softmax')
#     )  # 2 class with probability  0 or 1, for predict
#     model.summary()
#     # compile the model using Adam optimizer
#     model.compile(
#         loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy']
#     )
#     # train the model on training sets and validate on testing sets
#     model.fit(
#         x_train, y_train, batch_size=128, epochs=80, validation_data=(x_test, y_test)
#     )
#     # save the model
#     model.save('s2s')

#
# def generate_music(x_test, indexes_with_notes):
#     model = load_model('s2s')
#     index = np.random.randint(0, len(x_test) - 1)
#     # Get the data of generated index from x_test
#     music_pattern = x_test[index]
#     out_pred = []  # It will store predicted notes
#     # Iterate till 200 notes are generated
#     for i in range(200):
#         # Reshape the music pattern
#         music_pattern = music_pattern.reshape(1, len(music_pattern), 1)
#         # Get the maximum probability value from the predicted output
#         pred_index = np.argmax(model.predict(music_pattern))
#         # Get the note using predicted index and append to the output prediction list
#         out_pred.append(indexes_with_notes[pred_index])
#         music_pattern = np.append(music_pattern, pred_index)
#         # Update the music pattern with one timestep ahead
#         music_pattern = music_pattern[1:]
#     output_notes = []
#     for offset, pattern in enumerate(out_pred):
#         # If pattern is a chord instance
#         if ('.' in pattern) or pattern.isdigit():
#             # Split notes from the chord
#             notes_in_chord = pattern.split('.')
#             notes = []
#             for current_note in notes_in_chord:
#                 i_curr_note = int(current_note)
#                 new_note = note.Note(i_curr_note)
#                 new_note.storedInstrument = instrument.Piano()
#                 notes.append(new_note)
#             new_chord = chord.Chord(notes)
#             new_chord.offset = offset
#             output_notes.append(new_chord)
#         else:
#             new_note = note.Note(pattern)
#             new_note.offset = offset
#             new_note.storedInstrument = instrument.Piano()
#             output_notes.append(new_note)
#     midi_stream = stream.Stream(output_notes)
#     midi_stream.write('midi', fp='pred_music.mid')


if __name__ == '__main__':
    notes = []
    all_notes = []
    notes_per_file = []

    # Reading and extrating data
    music_processor = MusicProcessor(r'../data/schumann')
    midi_files = music_processor.get_midi_files()
    audio_streams = [music_processor.convert_midi_to_audio_stream(midi_file) for midi_file in midi_files]
    for audio_stream in audio_streams:
        notes = music_processor.extract_piano_notes(audio_stream)
        all_notes.extend(notes)
        notes_per_file.append(notes)
    # midi_files = get_midi_files(r'../data/schumann')
    # audio_stream = convert_midi_to_audio_stream(midi_files)
    # # if audio_stream is not None:
    # #     notes = extract_piano_notes(audio_stream)
    # #     all_notes.extend(notes)
    # #     notes_per_file.append(notes)
    # for midi_file in midi_files:
    #     notes = extract_piano_notes(audio_stream)
    #     notes_per_file.append(notes)
    #     all_notes.extend(notes)

    # Analysis and selection of data
    unique_notes = get_unique_notes(all_notes)
    notes_frequency = get_notes_frequency(unique_notes, all_notes)
    notes_frequency = get_notes_frequency_filtered_by_threshold(notes_frequency)
    filtered_notes_per_file = get_filtered_notes_by_frequency(
        notes_frequency, notes_per_file
    )
    indexes_with_notes = get_indexes_with_notes(notes_frequency)
    notes_with_indexes = get_notes_with_indexes(indexes_with_notes)

    # Data pre-processing
    x_new, y_new = create_input_output_arrays(
        filtered_notes_per_file, notes_with_indexes
    )
    x_train, x_test, y_train, y_test = get_train_test_sets(x_new, y_new)

    # Model
    create_model = MusicModel(notes_with_indexes)
    create_model.create_and_train_model(x_new)
    create_model.train_model(x_train, x_test, y_train, y_test)
    create_model.save_model()
    gen_music = NotesGenerator('s2s')
    gen_music.generate_music(x_test, indexes_with_notes)
