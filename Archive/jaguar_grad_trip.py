from proto import *
import itertools, random

rate = 44100

base_track = Track("Base", rate, 200.0)
track_bag = []
intro, vrs_chrs_1, vrs_chrs_2, bridge, vrs_3, end_chrss = base_track.top_beat.split([.5, 1.0, 1.05, .7, .5, 1.0])
vrs_1, chrs_1 = vrs_chrs_1.split_even(2)

def apply_rhythm(beat, rhythm_file, key_sound_map):
    with open(rhythm_file) as rf:
        char_times = eval(''.join(rf.readline()))
    beat_map = beat.interleave_split(char_times)
    for key, beats in beat_map.iteritems():
        for beat in beats:
            try:
                for sound, loc in key_sound_map[key]:
                    beat.attach(sound, loc)
            except:
                beat.attach(*key_sound_map[key])

def aulib(sound_dir):
    return os.path.join("audio", sound_dir)

def basic_rhythm(beat):
    clap_sound = RandomSound(rate)
    clap_sound.populate_with_dir(aulib("clap"))
    clave_sound = SpreadSound(rate, clap_sound, .1, .1, .01, 3)
    bass_sound = RandomSound(rate)
    bass_sound.populate_with_dir(aulib("bass_pulse"))
    accent_sound = RandomSound(rate)
    accent_sound.populate_with_dir(aulib("zona_blup"))
    key_sound_map = {
            'j': (clave_sound, (.5, .5)),
            'f': [(bass_sound, (-.5, .2)), (accent_sound, (-5, .5))],
            }
    apply_rhythm(beat, "one_and_and_x2.rh", key_sound_map)

def xylo_texture(beat):
    xylo = RandomPitchedSound(rate).populate_with_dir(aulib("g_major_xylo"))
    scale = "G,A,B,C,D,E,F".split(',')
    qwerty_xylo_map = {key:(xylo.for_pitch(pitch + random.choice(["_3", "_4"])), (1, i)) for key, pitch, i in zip("qwertyuiop", itertools.cycle(scale), itertools.count())}
    apply_rhythm(beat, "xylo_runs.rh", qwerty_xylo_map)

def zona_clink_drums(beat):
    clink = RandomPitchedSound(rate).populate_with_dir(aulib("zona_clink"))
    scale = "G,A,B,C,D,E,F".split(',')
    qwerty_clink_map = {key:(clink.for_pitch(pitch + "_3"), (8, i - 5)) for key, pitch, i in zip("qwertyuiop", itertools.cycle(scale), itertools.count())}
    apply_rhythm(beat, "zona_drums.rh", qwerty_clink_map)

def approaching_zona_roll(beat):
    bass_sound = RandomSound(rate)
    bass_sound.populate_with_dir(aulib("bass_pulse"))
    clink = RandomPitchedSound(rate).populate_with_dir(aulib("zona_clink"))
    scale = "G,A,B,C,D,E,F".split(',')
    qwerty_clink_map = {key:[(clink.for_pitch(pitch + "_3"), (8, -i*3 + 30)), (bass_sound, (.3, i*.2))] for key, pitch, i in zip("qwertyuiop", itertools.cycle(scale), itertools.count())}
    apply_rhythm(beat, "zona_drum_roll.rh", qwerty_clink_map)


def create_verse_basics(source_beat):
    verse_track = Track("Verse", rate, source_beat.duration(), source_beat.time())
    leadin, main1, main2 = verse_track.top_beat.split([1, 4.5, 4.5])
    approaching_zona_roll(leadin)
    meass = main1.split_even(4) + main2.split_even(4)
    for meas in meass:
        basic_rhythm(meas)

    clink_drums_track = Track("ZONG TONG ZONA BONG", rate, main1.duration() + main2.duration(), source_beat.time() + main1.time(), padding=.5)
    meass = clink_drums_track.top_beat.split_even(4)
    for meas in meass:
        zona_clink_drums(meas)

    tex_track = Track("Textures", rate, main2.duration(), source_beat.time() + main2.time(), padding=.5)
    random.seed(0)
    meass = tex_track.top_beat.split_even(4)
    for meas in meass:
        xylo_texture(meas)

    return [verse_track, clink_drums_track, tex_track]


track_bag = create_verse_basics(intro)
for track in track_bag:
    print track
mix = Mixer("Let's make some art, I guess...!", rate, track_bag)
mix.play()

