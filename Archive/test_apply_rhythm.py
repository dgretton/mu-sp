from proto import *

rate = 44100

base_track = Track("Base", rate, 40.0)
vrs_chrs_1, vrs_chrs_2, bridge, vrs_3, end_chrss = base_track.top_beat.split([1.0, 1.05, .7, .5, 1.0])
vrs_1, chrs_1 = vrs_chrs_1.split_even(2)

def apply_rhythm(beat, rhythm_file, key_sound_map):
    with open(rhythm_file) as rf:
        char_times = eval(''.join(rf.readline()))
    beat_map = beat.interleave_split(char_times)
    for key, beats in beat_map.iteritems():
        for beat in beats:
            beat.attach(*key_sound_map[key])

clave_sound = RandomSound(rate)
clave_sound.populate_with_dir(os.path.join("audio", "clap"))
bass_sound = RawSound(rate, os.path.join("audio", "hit", "hit.15.wav"))
accent_sound = RawSound(rate, os.path.join("audio", "hit", "clap.39.wav"))
key_sound_map = {'j': (clave_sound, (.5, .5)), 'f': (bass_sound, (-.5, .5)), 'k': (accent_sound, (6, 4))}

verse1_track = Track("Verse 1", rate, vrs_1.duration(), vrs_1.time())
apply_rhythm(verse1_track.top_beat, "save_rhythm.txt", key_sound_map)
mix = Mixer("Let's get applyin'!", rate, [verse1_track])
mix.play(quick_play=True)
