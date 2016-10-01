from proto import *

print "\"The Light,\" Sara Bareilles"
rate = 44100
skeleton = Track(rate, 4*60 + 24)
padding, intro, verse_1, verse_2, chorus_1, verse_3, chorus_2, outro, sustain = skeleton.top_beat.split([.01, .125, 1, 1, 1, 1, 1, .5, .5])
hit = randomSound(rate)
hit.populate_with_dir('audio\\TheLight\\bass_pulse')
hits = SpreadSound(rate, hit, 2, 2, .015, 3)
intro.attach(hit, (1, 1))
verse_1.attach(hit, (1, 1))
verse_2.attach(hit, (1, 1))
chorus_1.attach(hit, (1, 1))
verse_3.attach(hit, (1, 1))
chorus_2.attach(hit, (1, 1))
outro.attach(hit, (1, 1))
sustain.attach(hit, (1, 1))

The_Light = Mixer("The Light", rate, [skeleton])
The_Light.play(0.0, 10)