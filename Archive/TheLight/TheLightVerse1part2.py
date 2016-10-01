from proto import *
import itertools
import numpy as np

rate = 44100

scale = [None, "Eb", "F", "G", "Ab", "Bb", "C", "D"] # off by an index so e.g. [5] is the fifth

def scale_note(degree, octave):
    return scale[degree] + "_" + str(octave + 1 - (degree < 6))

verse1 = Track("Verse 1", rate, 70)
intro, sta nzas = verse1.top_beat.split([8, 30])

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

violin_triple_iter = itertools.cycle([ClippedSound(rate, random_violin.for_pitch(scale_note(1, 3)), .2, margin=.01),
                                    ClippedSound(rate, random_violin.for_pitch(scale_note(1, 3)), .2, margin=.01),
                                    ClippedSound(rate, random_violin.for_pitch(scale_note(2, 3)), .2, margin=.01)])

high_violin_source = RandomPitchedSound(rate, scale_note(1, 6)).populate_with_dir("audio\\TheLight\\bowed_violin")
high_violin = RandomIntervalSound(rate, ClippedSound(rate, high_violin_source, high_violin_source.duration()/2), .7, margin=.01)
vibrato_high_violin = ResampledSound(rate, high_violin, lambda x: np.sin(35.0*x)*.01+1)
spread_high_violin = SpreadSound(rate, vibrato_high_violin, 1, 2, 0, 1)

added_scratch = False
add_bass = False

for barpair_num, barpair, note, pent_notes, fifth_or_sixth in zip(itertools.count(), v1s1_barpairs, progression, pent_notes_for_progression, fifths_n_sixths * 2):
    for bar_num, bar in enumerate(barpair.split_even(2)):
        how_far_along = (barpair.time()-v1_stanza1.time())/v1_stanza1.duration()
        spread_string_chord = SpreadSound(rate, string_chord.for_pitch(note), .2, .2, .3 * (1-how_far_along), 5) # converge to alignment on beats
        bar.attach(ClippedSound(rate, spread_string_chord, barpair.duration() + .5), (2, 2))
        start_y = 12
        for half in bar.split_even(2):
            how_far_along = (half.time()-v1_stanza1.time())/v1_stanza1.duration()
            y_dist = start_y * (1 - np.sqrt(how_far_along))
            clipped_fifth_or_sixth = ClippedSound(rate, fifth_or_sixth, fifth_or_sixth.duration(), offset=0.01)
            pentuplet, triplet = half.split([.625, .375])
            if barpair_num != 7:
                pentuplet.attach(clipped_fifth_or_sixth, (-2, y_dist))
                pentuplet.attach(spread_high_violin, (10, 10 + y_dist*10))
            for beat, pent_note in zip(pentuplet.split([1, 1, 1, 1, .5]), pent_notes):
                #if barpair_num != 7:
                #    beat.attach(ClippedSound(rate, spread_string_chord, barpair.time() + barpair.duration() - beat.time() + .5), (2, 2))
                beat.attach(violin_triple_iter.next(), (3, 3))
                plucked_string = ClippedSound(rate, random_violin.for_pitch(scale_note(*pent_note)), beat.duration()*1.5, margin=.01)
                beat.attach(plucked_string, (.3, 1 + y_dist))
                if barpair_num >= 4:
                    degree, octave = pent_note
                    octave_string = ClippedSound(rate, random_violin.for_pitch(scale_note(degree, octave - 1)), beat.duration()*1.5, offset=.05)
                    beat.attach(octave_string, (1, 1 + y_dist))
            if barpair_num != 7:
                triplet.attach(clipped_fifth_or_sixth, (2, y_dist))
                triplet.attach(spread_high_violin, (-10, 10 + y_dist*10))
            for beat, trip_note in zip(triplet.split([1, 1, .5]), trip_notes):
                #if barpair_num != 7:
                #    beat.attach(ClippedSound(rate, spread_string_chord, barpair.time() + barpair.duration() - beat.time() + .3), (-2, 3))
                beat.attach(violin_triple_iter.next(), (3, 3))
                plucked_string = ClippedSound(rate, random_violin.for_pitch(scale_note(*trip_note)), beat.duration()*1.5, margin=.01)
                beat.attach(plucked_string, (.3, 1 + y_dist))
            if barpair_num == 7 and bar_num == 1:
                if add_bass:
                    half.attach(RawPitchedSound(rate, "audio\\TheLight\\bowed_violin_attack\\bowed_violin_attack_1.54.D_4.wav").for_pitch(scale_note(4, 2)), (-1, 1))
                if not added_scratch:
                    triplet.attach(ClippedSound(rate, RawPitchedSound(rate, "audio\\TheLight\\awful_scratchy_violin\\awful_scratchy_violin_3.40.B_3.wav").for_pitch(scale_note(4, 2)), triplet.duration()), (-1, 1))
                    added_scratch = True
                    add_bass = True


                
mix = Mixer("The Light", rate, [verse1])

mix.play_beat(v1_stanza1, quick_play=Truexml)
