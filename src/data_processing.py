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


if __name__ == '__main__':

    midi_files = read_midi_file(r'../All_Midi_Files/bach')
    all_notes = []
    for midi_file in midi_files:
        notes = extract_piano_notes(midi_file)
        all_notes.extend(notes)
    print(all_notes)
