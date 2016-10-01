from proto import *

rate = 44100

verse1 = Track(rate, 30)
intro, v1_stanza1, v1_stanza2 = verse1.top_beat.split([.15, .425, .425])

v1s1_halves = v1_stanza1.split_even(4)

for half in v1s1_halves:
    eighths = half.split_even(8)
    pattern = '15323154'
    pitch_code = {'1': "Eb_5", '2': "F_4", '3': "G_4", '4': "Ab_4", '5': "Bb_4"}
    for eighth, code in zip(eighths, pattern):
        pitch = pitch_code[code]
        xylo_hit = RandomPitchedSound(rate, pitch, []).populate_with_dir("audio\\TheLight\\G_major_xylo")
        xylo_hit = SpreadSound(rate, xylo_hit, .2, .2, 0, 1)
        eighth.attach(xylo_hit, (1, 2))

v1s2_quarters = v1_stanza2.split_even(64)

mix = Mixer("The Light", rate, [verse1])

mix.play()