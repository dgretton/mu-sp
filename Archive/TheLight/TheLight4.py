from proto import *
import itertools
import numpy as np

rate = 44100

scale = [None, "Eb", "F", "G", "Ab", "Bb", "C", "D"] # off by an index so e.g. [5] is the fifth

def scale_note(degree, octave):
    return scale[degree] + "_" + str(octave + 1 - (degree < 6))

verse1 = Track("Verse 1", rate, 70)
intro, stanzas = verse1.top_beat.split([8, 30])

v1_stanza1, v1_stanza2 = stanzas.split_even(2)

high_noise_raw = RawPitchedSound(rate, "audio\\TheLight\\noise_whine\\noise_whine.0.4200.wav").for_pitch(scale_note(1, 7))
long_high_noise = RandomIntervalSound(rate, high_noise_raw, v1_stanza1.duration(), margin=.2)
v1_stanza1.attach(long_high_noise, (5, 5))

v1s1_barpairs = v1_stanza1.split_even(8)

progression = [scale_note(1, 2), scale_note(4, 2), scale_note(5, 2), scale_note(1, 2),
                scale_note(6, 1), scale_note(4, 2), scale_note(5, 2), scale_note(7, 2)]
octave = 4
pent_notes_for_progression = ([[(1, octave + 1), (5, octave), (3, octave), (2, octave), (3, octave)],
                [(1, octave + 1), (5, octave), (4, octave), (3, octave), (4, octave)],
                [(1, octave + 1), (5, octave), (2, octave), (1, octave), (2, octave)],
                [(1, octave + 1), (5, octave), (3, octave), (2, octave), (3, octave)]] * 2)[:-1] + \
                [[(4, octave + 1), (2, octave + 1), (7, octave), (6, octave), (7, octave)]]
trip_notes = [(1, octave + 1), (5, octave), (4, octave)]

string_chord = RandomPitchedSound(rate).populate_with_dir("audio\\TheLight\\string_major_chord")
random_violin = RandomPitchedSound(rate).populate_with_dir("audio\\TheLight\\plucked_violin_damp")
sixth_pluck = RandomPitchedSound(rate).populate_with_dir("audio\\TheLight\\pluck_sixth").for_pitch(scale_note(1, 5))
fifth_pluck = RandomPitchedSound(rate).populate_with_dir("audio\\TheLight\\pluck_fifth").for_pitch(scale_note(1, 5))
fifths_n_sixths = [sixth_pluck, fifth_pluck, fifth_pluck, sixth_pluck]

high_violin_source = RandomPitchedSound(rate, scale_note(1, 6)).populate_with_dir("audio\\TheLight\\bowed_violin")
high_violin = RandomIntervalSound(rate, ClippedSound(rate, high_violin_source, high_violin_source.duration()/2), .7, margin=.01)
vibrato_high_violin = ResampledSound(rate, high_violin, lambda x: np.sin(35.0*x)*.01+1)
spread_high_violin = SpreadSound(rate, vibrato_high_violin, 1, 2, 0, 1)
spread_high_violin_2 = ResampledSound(rate, spread_high_violin, lambda x: 1.0 + (2.0**(-6) - 1.0)/(1 + np.exp((x - 5)*2)))

rasp_bass_source = RawPitchedSound(rate, "audio\\TheLight\\rasp_bass\\rasp_bass_1.0.110.wav")

for barpair_num, barpair, note, pent_notes, fifth_or_sixth in zip(itertools.count(), v1s1_barpairs, progression, pent_notes_for_progression, fifths_n_sixths * 2):
    if barpair_num >= 3:
        rasp_bass = RandomIntervalSound(rate, rasp_bass_source.for_pitch(note), barpair.duration())
        barpair.attach(rasp_bass, (.5, .5))
    for bar in barpair.split_even(2):
        how_far_along = (barpair.time()-v1_stanza1.time())/v1_stanza1.duration()
        spread_string_chord = SpreadSound(rate, string_chord.for_pitch(note), .2, .2, .5 * (1-how_far_along), 1) # converge to alignment on beats
        start_y = 2
        for half in bar.split_even(2):
            how_far_along = (half.time()-v1_stanza1.time())/v1_stanza1.duration()
            y_dist = start_y * (1 - np.sqrt(how_far_along))
            clipped_fifth_or_sixth = ClippedSound(rate, fifth_or_sixth, fifth_or_sixth.duration(), offset=0.01)
            pentuplet, triplet = half.split([625, 375])
            if barpair_num == 7:
                pentuplet.attach(spread_high_violin_2, (-10, 10 + y_dist*10))
            else:
                pentuplet.attach(clipped_fifth_or_sixth, (-2, y_dist))
                pentuplet.attach(spread_high_violin, (10, 10 + y_dist*10))
            for beat, pent_note in zip(pentuplet.split([1, 1, 1, 1, .5]), pent_notes):
                if barpair_num != 7:
                    beat.attach(ClippedSound(rate, spread_string_chord, barpair.time() + barpair.duration() - beat.time() + .5), (3, 3))
                plucked_string = ClippedSound(rate, random_violin.for_pitch(scale_note(*pent_note)), beat.duration()*1.5, margin=.01)
                beat.attach(plucked_string, (.3, 1 + y_dist))
                if barpair_num >= 4:
                    degree, octave = pent_note
                    octave_string = ClippedSound(rate, random_violin.for_pitch(scale_note(degree, octave - 1)), beat.duration()*1.5, offset=.05)
                    beat.attach(octave_string, (1, 1))
            if barpair_num == 7:
                triplet.attach(spread_high_violin_2, (-10, 10 + y_dist*10))
            else:
                triplet.attach(clipped_fifth_or_sixth, (2, y_dist))
                triplet.attach(spread_high_violin, (-10, 10 + y_dist*10))
            for beat, trip_note in zip(triplet.split([1, 1, .5]), trip_notes):
                if barpair_num != 7:
                    beat.attach(ClippedSound(rate, spread_string_chord, barpair.time() + barpair.duration() - beat.time() + .3), (-2, 3))
                plucked_string = ClippedSound(rate, random_violin.for_pitch(scale_note(*trip_note)), beat.duration()*1.5, margin=.01)
                beat.attach(plucked_string, (.3, 1 + y_dist))
                
mix = Mixer("The Light", rate, [verse1])

mix.play_beat(v1_stanza1, quick_play=True)