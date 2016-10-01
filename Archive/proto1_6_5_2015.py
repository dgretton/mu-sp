import wave
import winsound
import numpy as np
import scipy.io.wavfile as wavfile
import struct
from itertools import chain
import matplotlib.pyplot as plt

class Beat:
    
    def __init__(self, size=1.0, parent=None):
        self.size = size
        self.beats = []
        self.parent = parent
        self._duration = None
        self._time = None
        self.sounds = []
    
    def split(self, portions):
        if len(portions) < 2:
            print "NO NO NO can't split a beat into zero or one pieces"
            return
        if self.beats != []:
            print "NOPE NOPE NOPE can't split nonempty beat"
            return
        total = sum(portions)
        self.beats = [Beat(portion/total, self) for portion in portions]
        return self.beats
    
    def split_even(self, num):
        return self.split([1.0]*num)
    
    def duration(self):
        if self._duration is not None:
            return self._duration
        if self.parent is None:
            print "NO WAY JOSE the top level beat doesn't have a duration!"
            return None
        return self.parent.duration() * self.size
    
    def time(self):
        if self._time is not None:
            return self._time
        time = self.parent.time()
        if not isinstance(self.parent, Beat):
            return time
        for sibling_beat in self.parent.beats:
            if sibling_beat is self:
                break
            time += sibling_beat.duration()
        return time
    
    def attach(self, sound):
        self.sounds += [sound]
    
    def descendent_beats(self):
        if not self.beats:
            return [self]
        return [self] + list(chain(*[b.descendent_beats() for b in self.beats]))
        
class Sound:
    
    ear_separation = .2
    standard_distance = .2 # the distance at which a sound is imagined to be heard when it's at unit volume
    c_sound = 344.0
    
    def data(self):
        pass
    
    def duration(self):
        pass
    
    def _to_stereo(self, rate, mono_data, location):
        transform = np.tile(np.fft.rfft(mono_data), (2, 1))
        x, y = location
        left_dist = np.sqrt((x + Sound.ear_separation / 2)**2 + y*y)
        right_dist = np.sqrt((x - Sound.ear_separation / 2)**2 + y*y)
        # A shift of n is realized by a multiplication by exp(2pi*n*w/T) (but it can be fractional!)
        delays = np.array([[left_dist / Sound.c_sound], [right_dist / Sound.c_sound]])
        exp_coeff = 2j * np.pi * rate / len(mono_data)
        transformed = transform * np.exp(exp_coeff * delays * np.tile(np.arange(transform.shape[1]), (2, 1)))
        decays = np.array([[Sound.standard_distance/left_dist], [Sound.standard_distance/right_dist]])
        return np.fft.irfft(transformed) * decays
        

class RawSound(Sound):
    
    def __init__(self, rate, location, filename, registration_point=0.0):
        filerate, data = wavfile.read(filename)
        if filerate != rate:
            print "GOSH DARN IT the file "+filename+" has the wrong rate!"
        self.mono_data = np.array(data).astype(np.float) / 2**15
        self.reg_pt = registration_point
        self.duration = len(data) / float(rate)
        self.location = location
        self.rate = rate
    
    def data(self):
        return (self.reg_pt, self._to_stereo(self.rate, self.mono_data, self.location))
    
    def duration(self):
        return self.duration


class Track:
    
    def __init__(self, rate, duration, volume=1.0):
        self.rate = rate
        self.top_beat = Beat(parent=self)
        self._duration = duration
        self.volume = volume
        self.data = np.zeros((2, int(rate * duration) + 1))
            
    def mix_into(self, start_time, buffer):
        self._render()  
        start_index = int(start_time * self.rate)
        data_to_mix = self.data[:, start_index : start_index + buffer.shape[1]]
        return self._mix(data_to_mix * self.volume, buffer)
    
    def _mix(self, a, b):
        return a + b - a * b
    
    def duration(self):
        return self._duration
    
    def time(self):
        return 0.0
    
    def _render(self):
        for beat in self.top_beat.descendent_beats():
            for sound in beat.sounds:
                reg_pt, sound_data = sound.data()
                start_time = beat.time() - reg_pt
                if start_time < 0:
                    start_index = 0
                    sound_data = sound_data[:, int(start_time * self.rate) :]
                else:
                    start_index = int(start_time * self.rate)
                end_index = start_index + sound_data.shape[1]
                if end_index >= self.data.shape[1]:
                    end_index = self.data.shape[1] - 1
                    sound_data = sound_data[:, : end_index - start_index]
                self.data = np.concatenate((self.data[:, :start_index], self._mix(self.data[:, start_index : end_index], sound_data), self.data[:, end_index:]), axis=1)
        
        

class Work:
    
    def __init__(self, name, rate, tracks=[]):
        self.name = name
        self.tracks = tracks
        self.rate = rate
    
    def play(self, t0, t1):
        play_buffer = np.zeros((2, int(rate * (t1 - t0)) + 1))
        for track in self.tracks:
            play_buffer = track.mix_into(t0, play_buffer)
        play_buffer = (play_buffer * 2**15).astype(np.int16)
        
        temp_output = wave.open('temp.wav', 'w')
        temp_output.setparams((2, 2, self.rate, 0, 'NONE', 'not compressed')) # (nchannels, samplewidth, framerate, nframes, compressiontype, compressionname)
        raw_play_data = []
        for left_sample, right_sample in zip(play_buffer[0], play_buffer[1]):
            left_bytes = struct.pack('h', left_sample)
            right_bytes = struct.pack('h', right_sample)
            raw_play_data.append(left_bytes)
            raw_play_data.append(right_bytes)
        temp_output.writeframes(''.join(raw_play_data))
        temp_output.close()
        winsound.PlaySound('temp.wav', winsound.SND_ALIAS)
    

if __name__ == "__main__":
    print "TEST: let's make a beeeat! wee!"
    rate = 44100
    track1 = Track(rate, 5)
    quarters = track1.top_beat.split_even(8)
    hits = [RawSound(rate, (i*.2, -1.0), 'audio\clap.wav') for i in range(-15, 15)]
    for i, quarter in enumerate(quarters):
        diddle = quarter.split_even(i % 4 + 2)
        for note in diddle:
            note.attach(hits.pop())
    #print [b.time() for b in track1.top_beat.descendent_beats()]
    testWork = Work("Da Bomba Beat", rate, [track1])
    testWork.play(0.0, 4.9)
    