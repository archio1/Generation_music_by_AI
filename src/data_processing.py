import os
from music21 import note, chord, instrument, converter, stream
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dropout, Dense
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
    datefmt='%m-%d %H:%M',
    filename='logs/data_processing.log',
    filemode='w')

# This function takes a folder path as input and reads all MIDI files in it
# It returns a list of audio streams for each MIDI file
def read_midi_file(folder_path):
    audio_streams = []
    if os.path.exists(folder_path):
        for dir_path, dir_names, filenames in os.walk(folder_path):
            for filename in filenames:
                # check if the file ends with ".mid" or ".midi"
                if filename.endswith('.mid') or filename.endswith('.midi'):
                    # read the MIDI file and append the audio stream to the list
                    try:
                        audio_stream = converter.parse(os.path.join(dir_path, filename))
                        audio_streams.append(audio_stream)
                    except KeyError:
                        logging.debug(f'[KeyError] Fail of converting for {filename}')
    else:
        print(f'Current path not found: \n{os.path.abspath(folder_path)} ')
        exit()
    return audio_streams


# This function takes a list of audio streams as input and extracts piano notes from them
# It returns a list of all the notes found in the audio streams
def extract_piano_notes(audio_streams):
    notes = []
    # separate the audio streams by instrument
    instrmt = instrument.partitionByInstrument(audio_streams)
    for part in instrmt.parts:
        # check if the instrument is piano
        if 'Piano' in str(part):
            # get all the notes from the audio stream
            notes_to_parse = part.recurse()
            for element in notes_to_parse:
                # check if the element is a Note object
                if isinstance(element, note.Note):
                    # append the pitch of the note to the notes list
                    notes.append(str(element.pitch))
                # check if the element is a Chord object
                elif isinstance(element, chord.Chord):
                    # append the pitches of the chord to the notes list
                    notes.append('.'.join(str(n) for n in element.normalOrder))
    return notes


def get_unique_notes(notes_array):
    # Create set of unique notes - non-repeating notes

    unique_notes = list(set(notes_array))
    print("Unique Notes:", len(unique_notes))

    return unique_notes


def get_note_freq(unique_notes, notes_array):
    # Create a dictionary with the frequency of each note in the input array

    freq = dict(map(lambda x: (x, notes_array.count(x)), unique_notes))

    return freq


def get_freq_threshold(freq):
    # Determine the number of notes whose frequency is above a certain threshold
    # for different threshold values (30, 50, 70, and 90)

    for i in range(30, 100, 20):
        print(i, ":", len(list(filter(lambda x: x[1] >= i, freq.items()))))

    # Create a dictionary with notes whose frequency is above 50
    freq_notes = dict(filter(lambda x: x[1] >= 50, freq.items()))

    return freq_notes


def create_new_notes(freq_notes, notes_array):
    # Create a new array of notes that includes only the notes with frequency above 50

    new_notes = [[i for i in j if i in freq_notes] for j in notes_array]
    return new_notes


def create_ind2note_dict(freq_notes):
    # Create a dictionary with keys as note indices and values as notes themselves

    ind2note = dict(enumerate(freq_notes))
    return ind2note


def create_note2ind_dict(ind2note):
    # Create inverted dictionary with keys as notes themselves and values as note indices

    note2ind = dict(map(reversed, ind2note.items()))
    return note2ind


def create_input_output_arrays(new_notes, note2ind, timesteps=50):
    x = []
    y = []
    for i in new_notes:
        for j in range(0, len(i) - timesteps):
            inp = i[j:j + timesteps]
            out = i[j + timesteps]
            x.append(list(map(lambda x: note2ind[x], inp)))
            y.append(note2ind[out])
    x_new = np.array(x)
    y_new = np.array(y)
    x_new = np.reshape(x_new, (len(x_new), timesteps, 1))
    y_new = np.reshape(y_new, (-1, 1))
    return x_new, y_new


def get_train_test_sets(x_new, y_new, timesteps=50):

    x_train, x_test, y_train, y_test = train_test_split(x_new, y_new, test_size=0.2, random_state=42)

    return x_train, x_test, y_train, y_test


def create_and_train_model(x_new, note2ind, x_train, x_test, y_train, y_test):
    # Why is this type of model?
    model = Sequential()
    # 2 layer LSTM?
    model.add(LSTM(256, return_sequences=True, input_shape=(x_new.shape[1], x_new.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(256))
    model.add(Dropout(0.2))
    # This activation function?
    model.add(Dense(256, activation='relu')) # f(x) = max(0, x)
    model.add(Dense(len(note2ind), activation='softmax'))# 2 class with probability  0 or 1, for predict
    model.summary()
    # compile the model using Adam optimizer
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # train the model on training sets and validate on testing sets
    model.fit(x_train, y_train, batch_size=128, epochs=80, validation_data=(x_test, y_test))
    # save the model
    model.save("s2s")


def generate_music(x_test, ind2note):
    model = load_model('s2s')
    index = np.random.randint(0, len(x_test) - 1)
    # Get the data of generated index from x_test
    music_pattern = x_test[index]
    out_pred = []  # It will store predicted notes
    # Iterate till 200 notes are generated
    for i in range(200):
        # Reshape the music pattern
        music_pattern = music_pattern.reshape(1, len(music_pattern), 1)
        # Get the maximum probability value from the predicted output
        pred_index = np.argmax(model.predict(music_pattern))
        # Get the note using predicted index and append to the output prediction list
        out_pred.append(ind2note[pred_index])
        music_pattern = np.append(music_pattern, pred_index)
        # Update the music pattern with one timestep ahead
        music_pattern = music_pattern[1:]
    output_notes = []
    for offset, pattern in enumerate(out_pred):
        # If pattern is a chord instance
        if ('.' in pattern) or pattern.isdigit():
            # Split notes from the chord
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                i_curr_note = int(current_note)
                new_note = note.Note(i_curr_note)
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)
    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp='pred_music.mid')

if __name__ == '__main__':
    midi_files = read_midi_file(r'../data/schumann')
    all_notes = []
    notes_per_one = []
    for midi_file in midi_files:
        notes = extract_piano_notes(midi_file)
        all_notes.extend(notes)#append
    for midi_file in midi_files:
        notes = extract_piano_notes(midi_file)
        notes_per_one.append(notes)

    unique_notes = get_unique_notes(all_notes)
    frequency = get_note_freq(unique_notes, all_notes)
    freq_notes = get_freq_threshold(frequency)
    new_notes = create_new_notes(freq_notes, notes_per_one)
    ind2note = create_ind2note_dict(freq_notes)
    note2ind = create_note2ind_dict(ind2note)
    x_new, y_new = create_input_output_arrays(new_notes, note2ind)
    x_train, x_test, y_train, y_test = get_train_test_sets(x_new, y_new)
    create_and_train_model(x_new, note2ind, x_train, x_test, y_train, y_test)
    generate_music(x_test, ind2note)
