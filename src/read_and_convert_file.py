import logging
import json
from typing import List
from music21 import converter, instrument, note, chord, stream
from pathlib import Path

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
    datefmt='%m-%d %H:%M',
    filename='data_processing.log',
    filemode='w',
)


class MusicProcessor:
    def __init__(self, folder_path):
        self.folder_path = Path(folder_path)
        if not self.folder_path.exists():
            print(f'Current path not found: \n{self.folder_path.absolute()} ')
            exit()

    def get_midi_files(self):
        midi_files = []
        for file_path in self.folder_path.rglob('**/*.mid'):
            midi_files.append(file_path)
        return midi_files

    def convert_midi_to_audio_stream(self, midi_file):
        try:
            audio_stream = converter.parse(midi_file)
            print('Parsing %s' % midi_file)
            return audio_stream
        except KeyError:
            logging.debug(f'[KeyError] Fail of converting for {midi_file}')
            return None

    def extract_piano_notes(self, audio_stream: stream.Score) -> List[str]:
        """Takes an audio stream and extracts piano notes from it.

        Args:
            audio_stream (stream.Score): container of music21 objects

        Returns:
            List[str]: converted list of notes
        """
        notes = []
        offsets = []
        durations = []
        notes_to_parse = None

        # get stream grouped by each unique instrument
        audio_stream = instrument.partitionByInstrument(audio_stream)
        for part in audio_stream.parts:
            if 'Piano' in str(part):
                # get elements (notes, chords, tempos, meters ...) from stream
                part_elements = part.recurse()
                offset_base = 0
                for element in part_elements:
                    if isinstance(element, note.Note):
                        notes.append(str(element.pitch))
                        durations.append(str(element.duration.quarterLength))
                        offsets.append(str(element.offset - offset_base))
                    elif isinstance(element, chord.Chord):
                        # append the pitches of the chord to the notes list
                        notes.append(
                            '.'.join(str(notes) for notes in element.normalOrder)
                        )
                        durations.append(str(element.duration.quarterLength))
                        offsets.append(str(element.offset - offset_base))
                    offset_base = element.offset

        with open('../data/notes.json', 'w') as f:
            json.dump(notes, f, indent=4)

        with open('../data/durations.json', 'w') as f:
            json.dump(durations, f, indent=4)

        with open('../data/offsets.json', 'w') as f:
            json.dump(offsets, f, indent=4)

        return notes
