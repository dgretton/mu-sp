from proto import *
from math import sin, cos, pi
import itertools, random

rate = Sound.default_rate
quarter_duration = .61
quarter_duration_slower = .8

base_track = Track("Base", rate)
track_bag = []
verse1, comment1, verse2, comment2, ec1, ec2, ec3, ec4, ec5, ec6 = base_track.top_beat.split(10)
endclaves = [ec1, ec2, ec3, ec4, ec5, ec6]

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

def rhlib(rh_name):
    return rh_name + ".rh"

def loctrans(far, angle):
    return (far*sin(angle), far*cos(angle))

clap_sound = RandomSound()
clap_sound.populate_with_dir(aulib("clap"))
claps_sound = SpreadSound(clap_sound, .02, .02, .01, 3)

def clap(beat, claps_sound=claps_sound):
    beat.attach(claps_sound, loctrans(.3, random.random()*2*pi))

def clapclap(beat):
    cl1, cl2 = beat.split_even(2)
    clap(cl1)
    clap(cl2)

bass_sound = RandomSound()
bass_sound.populate_with_dir(aulib("bass_pulse"))

xylo = RandomPitchedSound().populate_with_dir(aulib("g_major_xylo"))
clave_sound = ClippedSound(xylo.for_pitch("D_6"), .05)

def clave_in_five(beat, sound=clave_sound):
    clave_sound = sound
    key_sound_map = {
           'f': (clave_sound, (1, 1)),
           }
    apply_rhythm(beat, rhlib("clave_in_five"), key_sound_map)


rasp_source = RawPitchedSound(os.path.join(aulib("rasp_bass"), "rasp_bass_1.0.110.wav"))
rasps = {key:RandomIntervalSound(rasp_source.for_pitch(note),
    quarter_duration*3/4) for key, note in zip(["high", "med", "low"], \
            ["G_2", "F#_2", "C_2"])}

def create_comment(track, beat, sounds=(bass_sound, clave_sound, \
        xylo, rasps)):
    bass_sound, clave_sound, xylo, rasps = sounds
    comment_main_track = Track("Comment-Main", rate)
    root_beat = comment_main_track.link_root(beat, track)
    plink_lead_in, all_lines = root_beat.split(2)
    linepairs = all_lines.split_even(4)

    rasp_track = Track("Comment-Rasp-Bass", rate)
    rasp_root_beat = rasp_track.link_root(all_lines, comment_main_track)

    print "############ FIRST SET DURATION ##############"
    plink_lead_in.set_duration(quarter_duration * 4)

    for n, plink, pitch in zip(range(4), plink_lead_in.split_even(4), 
            ["G_4", "D_4", "F_4", "G#_3"]):
        plink.attach(xylo.for_pitch(pitch), loctrans(4-n, pi/3*(n-2)))

    skipped_first_claps = False
    for lp, rasp_quarter in zip(linepairs, rasp_root_beat.split_even(4)):
        sixteenline, fiveline = lp.split(2)

        sixteen_beats = sixteenline.split_even(16)
        for i_of_16, b_16 in enumerate(sixteen_beats):
            if skipped_first_claps:
                if i_of_16 % 4 == 1:
                    clapclap(b_16)
                else:
                    clap(b_16)
            b_16.attach(bass_sound, loctrans(.3, pi/4))
        skipped_first_claps = True
        print "############ SECOND SET DURATION ##############"
        sixteen_beats[0].set_duration(quarter_duration/2)
        
        rasp_8_portion, slack = rasp_quarter.split(2)
        rasp_8_portion.link(sixteenline)
        rasp_beats = rasp_8_portion.split([3, 3, 2 + 8, 3, 3, 2 + 8])
        rasp_loc = loctrans(.9, -pi*.48)
        for rasp_beat, key in zip(rasp_beats, ["high", "med", "low",
                "med", "high", "low"]):
            rasp_beat.attach(rasps[key], rasp_loc)

        fiveline.set_duration(quarter_duration*5)
        clave_in_five(fiveline)
        # five_beats = fiveline.split_even(5)
        # for b_5 in five_beats:
        #     b_5.attach(clave_sound, loctrans(4, -pi/4))
        # five_beats[0].set_duration(quarter_duration)

    return [comment_main_track, rasp_track]

#def verse_rhythm(beat):
#    key_sound_map = {
#            'j': (clave_sound, (.5, .5)),
#            'f': [(bass_sound, (-.5, .5)), (accent_sound, (-5, .5))],
#            }
#    apply_rhythm(beat, rhlib("one_and_and_x2"), key_sound_map)

#def xylo_texture(beat, sounds=[xylo]):
    #xylo, = sounds
    #scale = "G,A,B,C,D,E,F".split(',')
    #qwerty_xylo_map = {key:(xylo.for_pitch(pitch + random.choice(["_3", "_4"])), (1, i)) for key, pitch, i in zip("qwertyuiop", itertools.cycle(scale), itertools.count())}
    #apply_rhythm(beat, rhlib("xylo_runs"), qwerty_xylo_map)

def zona_clink_drums(beat):
    clink = RandomPitchedSound().populate_with_dir(aulib("zona_clink"))
    scale = "G,A,B,C,D,E,F".split(',')
    qwerty_clink_map = {key:(clink.for_pitch(pitch + "_3"), (8, i - 5)) for key, pitch, i in zip("qwertyuiop", itertools.cycle(scale), itertools.count())}
    apply_rhythm(beat, rhlib("zona_drums"), qwerty_clink_map)

def approaching_zona_roll(beat):
    bass_sound = RandomSound()
    bass_sound.populate_with_dir(aulib("bass_pulse"))
    clink = RandomPitchedSound().populate_with_dir(aulib("zona_clink"))
    scale = "G,A,B,C,D,E,F".split(',')
    qwerty_clink_map = {key:[(clink.for_pitch(pitch + "_3"), (8, -i*3 + 30)), (bass_sound, (.3, i*.2))] for key, pitch, i in zip("qwertyuiop", itertools.cycle(scale), itertools.count())}
    apply_rhythm(beat, rhlib("zona_drum_roll"), qwerty_clink_map)


def create_verse(track, beat, sounds=(bass_sound, clave_sound, \
        xylo, rasps)):
    bass_sound, clave_sound, xylo, rasps = sounds
    verse_track = Track("Verse-Main", rate)
    root_beat = verse_track.link_root(beat, track)

    leadin, proper = root_beat.split(2)
    approaching_zona_roll(leadin)
    leadin.set_duration(quarter_duration_slower*2)
    for line in proper.split_even(4):
        line.set_duration(quarter_duration_slower*8)
        zona_clink_drums(line)
    return [verse_track]

#    meass = main1.split_even(4) + main2.split_even(4)
#    for meas in meass:
#        basic_rhythm(meas)
#
#    clink_drums_track = Track("ZONG TONG ZONA BONG", rate, main1.duration() + main2.duration(), source_beat.time() + main1.time(), padding=.5)
#    meass = clink_drums_track.top_beat.split_even(4)
#    for meas in meass:
#        zona_clink_drums(meas)
#
#    tex_track = Track("Textures", rate, main2.duration(), source_beat.time() + main2.time(), padding=.5)
#    random.seed(0)
#    meass = tex_track.top_beat.split_even(4)
#    for meas in meass:
#        xylo_texture(meas)
#
#    return [verse_track, clink_drums_track, tex_track]


track_bag = create_verse(base_track, verse1) + \
        create_comment(base_track, comment1) + \
        create_verse(base_track, verse2) + \
        create_comment(base_track, comment2)

for end_clave in endclaves:
    clave_track = Track("Outtro-Claves", rate)
    root_beat = clave_track.link_root(end_clave, base_track)
    clave_in_five(end_clave)
    end_clave.set_duration(quarter_duration*5)
    track_bag.append(clave_track)

for track in track_bag:
    print track
mix = Mixer("Let's make some art, I guess...!", rate, track_bag)
mix.play()

