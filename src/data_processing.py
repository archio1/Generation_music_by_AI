import os
from music21 import note, chord, instrument, converter


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
                    audio_stream = converter.parse(os.path.join(dir_path, filename))
                    audio_streams.append(audio_stream)
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


if __name__ == '__main__':

    midi_files = read_midi_file(r'../data/midi/bach')
    all_notes = []
    for midi_file in midi_files:
        notes = extract_piano_notes(midi_file)
        all_notes.extend(notes)
    print(all_notes)
    unique_notes = get_unique_notes(all_notes)
    frequency = get_note_freq(unique_notes, all_notes)
    freq_notes = get_freq_threshold(frequency)
    new_notes = create_new_notes(freq_notes, all_notes)
    ind2note = create_ind2note_dict(freq_notes)
    note2ind = create_note2ind_dict(ind2note)
