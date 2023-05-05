from music21 import instrument, note, chord, stream
import numpy as np
from keras.models import load_model


class NotesGenerator:

    def __init__(self, model_path):
        self.model = load_model(model_path)

    def generate_music(self, x_test, indexes_with_notes):
        index = np.random.randint(0, len(x_test) - 1)
        # Get the data of generated index from x_test
        music_pattern = x_test[index]
        out_pred = []  # It will store predicted notes
        # Iterate till 200 notes are generated
        for i in range(200):
            # Reshape the music pattern
            music_pattern = music_pattern.reshape(1, len(music_pattern), 1)
            # Get the maximum probability value from the predicted output
            pred_index = np.argmax(self.model.predict(music_pattern))
            # Get the note using predicted index and append to the output prediction list
            out_pred.append(indexes_with_notes[pred_index])
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