from proto import *
import itertools
import numpy as np

rate = 44100

scale = [None, "Eb", "F", "G", "Ab", "Bb", "C", "D"] # off by an index so e.g. [5] is the fifth

def scale_note(degree, octave):
    return scale[degree] + "_" + str(octave + 1 - (degree < 6))


#scale_test = Track("Scale", rate, 5)
#bump = scale_test.top_beat
#for i, beat in enumerate(bump.split_even(8)):
#    xylo_hit = RandomPitchedSound(rate, scale_note(i%7 + 1, 4 + (i==7))).populate_with_dir("audio\\TheLight\\G_major_xylo")
#    xylo_hit = SpreadSound(rate, xylo_hit, .2, .2, 0, 1)
#    beat.attach(xylo_hit, (1, 2))
#
#mix = Mixer("doopadoop", rate, [scale_test])
#mix.play()
#exit()

#pop_test = Track("pops", rate, 5, 5.5)
#bump = pop_test.top_beat
#for i, beat in enumerate(bump.split_even(8)):
#    xylo_hit = RawSound(rate, "audio\\TheLight\\G_major_xylo\\G_major_xylo_Bb.23.B_4.wav")
#    beat.attach(xylo_hit, (.5*(i-4), 1))
#
#mix = Mixer("doopadoop", rate, [pop_test])
#mix.play()
#exit()

verse1 = Track("Verse 1", rate, 38)
intro, v1_stanza1, v1_stanza2 = verse1.top_beat.split([.15, .425, .425])

v1s1_halves = v1_stanza1.split_even(16)

progression = [scale_note(1, 2), scale_note(4, 2), scale_note(5, 2), scale_note(1, 2)]

for i, half in enumerate(v1s1_halves):
    how_far_along = (half.time()-v1_stanza1.time())/v1_stanza1.duration()
    if i % 4 == 0:
        pitch = progression.pop(0)
        string_chord = RandomPitchedSound(rate, pitch).populate_with_dir("audio\\TheLight\\string_major_chord")
        string_chord = SpreadSound(rate, string_chord, .2, .2, .5 * (1-how_far_along), 2) # converge to alignment on beats
    half.attach(string_chord, (-2, 0))
    if i == 15:
        string_bass_note = RandomPitchedSound(rate, scale_note(7, 1)).populate_with_dir("audio\\TheLight\\bowed_violin")
        string_bass_lead = SpreadSound(rate, string_bass_note, .2, .2, 0, 1)
        sized_string_bass = ClippedSound(rate, string_bass_lead, half.duration())
        half.attach(sized_string_bass, (0, 2))

v1s1_quarters = itertools.chain(*[half.split([.375, .825]) for half in v1s1_halves]) # slight gallop

high_violin = RandomPitchedSound(rate, scale_note(1, 6)).populate_with_dir("audio\\TheLight\\bowed_violin")
high_violin = SpreadSound(rate, high_violin, 3, 5, 0, 1)
high_violin_atk = RandomPitchedSound(rate, scale_note(1, 6)).populate_with_dir("audio\\TheLight\\bowed_violin_attack")
high_violin_atk = SpreadSound(rate, high_violin_atk, 3, 5, 0, 1)
for quarter in v1s1_quarters:
    quarter.attach(high_violin_atk, (18, 4))
    for sixteenth in quarter.split_even(4)[1:]:
        sixteenth.attach(high_violin, (18, 4))


#string_drone = Track("String Drone", rate, v1_stanza1.duration(), v1_stanza1.time())
#random_strings = string_drone.top_beat.split([np.random.random() for i in range(40)])
## drop = RandomSound(rate, scale_note(1, 3)).populate_with_dir("audio\\TheLight\\water_drop_reverb")
#string = RandomPitchedSound(rate, scale_note(1, 3)).populate_with_dir("audio\\TheLight\\string_major_chord")
#string = SpreadSound(rate, string, 1, 1, .2, 1)
#for string_beat in random_strings:
#    how_far_along = (string_beat.time())/string_drone.duration()
#    string_beat.attach(string, (2.5, 5 - how_far_along * 10))
#    string_beat.attach(string, (-2.5, 5 - how_far_along * 10))
#    #if np.random.random() < how_far_along:
#    #    string_beat.attach(drop, (-.5, .5))



v1s2_halves = v1_stanza2.split_even(16) # ([np.sin((n-5)/11.0*np.pi) + 10 for n in range(16)])

string_chord_up = RandomPitchedSound(rate, scale_note(1, 3)).populate_with_dir("audio\\TheLight\\string_major_chord")
string_chord_up = SpreadSound(rate, string_chord_up, .2, .2, 0, 2)
xylo_hit = RandomPitchedSound(rate, scale_note(1, 5)).populate_with_dir("audio\\TheLight\\G_major_xylo")
xylo_hit = SpreadSound(rate, xylo_hit, .2, .2, 0, 4)
xylo_hit_up = RandomPitchedSound(rate, scale_note(1, 6)).populate_with_dir("audio\\TheLight\\G_major_xylo")
xylo_hit_up = SpreadSound(rate, xylo_hit_up, .2, .2, 0, 1)
      
progression = [scale_note(6, 1), scale_note(2, 2), scale_note(5, 2), scale_note(7, 1)]

for i, half in enumerate(v1s2_halves):
    if i % 4 == 0:
        pitch = progression.pop(0)
        string_chord = RandomPitchedSound(rate, pitch).populate_with_dir("audio\\TheLight\\string_major_chord")
        string_chord = SpreadSound(rate, string_chord, .2, .2, 0, 1)
        if i == 4:
            pitch = scale_note(2, 1)
        if i == 8:
            pitch = scale_note(5, 1)
        string_bass = RandomPitchedSound(rate, pitch).populate_with_dir("audio\\TheLight\\bowed_violin")
        string_bass = SpreadSound(rate, string_bass, .2, .2, 0, 1)
        half.attach(string_bass, (0, 2))
    if i % 2 == 0:
        half.attach(string_chord, (-2, 0))
    half.attach(string_chord_up, (2, 0))
    half.attach(xylo_hit_up, (1, 5))
    quarters = half.split_even(4)
    for quarter in quarters:
        quarter.attach(xylo_hit, (-1, 5))


mix = Mixer("The Light", rate, [verse1])

mix.play()#_beat(v1_stanza2)