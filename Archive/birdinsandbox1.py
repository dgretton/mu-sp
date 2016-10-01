from proto import *

rate = 44100

base_track = Track("Base", rate, 40.0)
vrs_chrs_1, vrs_chrs_2, bridge, vrs_3, end_chrss = base_track.top_beat.split([1.0, 1.05, .7, .5, 1.0])
vrs_1, chrs_1 = vrs_chrs_1.split_even(2)

def pseudo_latin(beat, repeats):
    clave_sound = RandomSound(rate)
    clave_sound.populate_with_dir(os.path.join("audio", "clap"))
    bass_sound = RawSound(rate, os.path.join("audio", "hit", "hit.15.wav"))
    accent_sound = RawSound(rate, os.path.join("audio", "hit", "clap.39.wav"))
    for rep in beat.split_even(repeats):
        clave_pos = [0, 2, 5, 7, 9, 11, 13]
        bass_pos = [0, 3, 4, 8, 12, 15]
        accent_pos = [2, 6, 10, 14]
        for i, latin_beat in enumerate(rep.split_even(16)):
            if i in clave_pos:
                latin_beat.attach(clave_sound, (.5, .5))
            if i in bass_pos:
                latin_beat.attach(bass_sound, (-.5, .5))
            if i in accent_pos:
                latin_beat.attach(accent_sound, (6, 4))
            
def create_verse(source_beat):
    verse_track = Track("Verse", rate, source_beat.duration(), source_beat.time())
    pseudo_latin(verse_track.top_beat, 2)
    return verse_track

mix = Mixer("Let's Get \'Latin!\'", rate, [create_verse(vrs_1)])
mix.play(quick_play=False)
