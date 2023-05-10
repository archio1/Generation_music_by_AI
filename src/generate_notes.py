from music21 import instrument, note, chord, stream
import numpy as np
from keras.models import load_model


class NotesGenerator:

    def __init__(self, model_path):
        self.model = load_model(model_path)

    def generate_notes(self, network_input, pitchnames, n_vocab):
        start = np.random.randint(0, len(network_input) - 1)

        int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

        pattern = network_input[start]
        prediction_output = []

        # generate 500 notes
        for note_index in range(500):
            prediction_input = np.reshape(pattern, (1, len(pattern), 1))
            prediction_input = prediction_input / float(n_vocab)
            prediction = self.model.predict(prediction_input, verbose=0)
            index = np.argmax(prediction)
            result = int_to_note[index]
            prediction_output.append(result)
            pattern = np.append(pattern, index)
            pattern = pattern[1:len(pattern)]

        return prediction_output

    def create_midi(self, prediction_output):
        offset = 0
        output_notes = []

        # create note and chord objects based on the values generated by the model
        for pattern in prediction_output:
            # pattern is a chord
            if ('.' in pattern) or pattern.isdigit():
                notes_in_chord = pattern.split('.')
                notes = []
                for current_note in notes_in_chord:
                    new_note = note.Note(int(current_note))
                    new_note.storedInstrument = instrument.Piano()
                    notes.append(new_note)
                new_chord = chord.Chord(notes)
                new_chord.offset = offset
                output_notes.append(new_chord)
            # pattern is a note
            else:
                new_note = note.Note(pattern)
                new_note.offset = offset
                new_note.storedInstrument = instrument.Piano()
                output_notes.append(new_note)

            # increase offset each iteration so that notes do not stack
            offset += 0.5

        midi_stream = stream.Stream(output_notes)
        midi_stream.write('midi', fp='test_output.mid')
