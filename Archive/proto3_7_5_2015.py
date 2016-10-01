import wave, winsound, struct, os, math
import numpy as np
import scipy.io.wavfile as wavfile
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
    
    def attach(self, sound, location):
        self.sounds += [(sound, location)]
    
    def descendent_beats(self):
        if not self.beats:
            return [self]
        return [self] + list(chain(*[b.descendent_beats() for b in self.beats]))
        
class Sound:
    
    ear_separation = .2
    standard_distance = .2 # the distance at which a sound is imagined to be heard when it's at unit volume
    c_sound = 344.0
    note_map = {name: index for index, name in enumerate(["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"])}
    note_map.update([("Bb", 1), ("Db", 4), ("Eb", 6), ("Gb", 9), ("Ab", 11)])
    temper_ratio = 2.0**(1.0/12)
    
    def render_from(self, location):
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
    
    def _read_mono_data(self, rate, filename):
        filerate, data = wavfile.read(filename)
        if filerate != rate:
            print "GOSH DARN IT the file "+filename+" has the wrong rate!"
        return np.array(data).astype(np.float) / 2**15
    
    @staticmethod
    def note_frequency(name, octave=4):
        scale_index = Sound.note_map[name]
        return 440 * Sound.temper_ratio**scale_index * 2**(octave - 4 - (scale_index % 12 > 2))

class RawSound(Sound):
    
    def __init__(self, rate, file_path, registration_point=None):
        if registration_point is not None:
            self.reg_pt = registration_point
        else:
            parse = os.path.basename(file_path).split('.')
            if len(parse) > 2:
                self.reg_pt = float(parse[1])/1000 # field after first dot in file name is reg_pt ms
            else:
                self.reg_pt = 0.0
        self.mono_data = self._read_mono_data(rate, file_path)
        self.rate = rate
    
    def render_from(self, location):
        return (self.reg_pt, self._to_stereo(self.rate, self.mono_data, location))
    
    def duration(self):
        return len(self.mono_data) / float(rate)

class RandomSound(Sound):
    
    def __init__(self, rate, sounds=[]):
        self.sounds = sounds
        self.rate = rate
    
    def render_from(self, location):
        random_sound = self.sounds[np.random.randint(len(self.sounds))]
        return random_sound.render_from(location)
        
    def duration(self):
        return max([snd.duration() for snd in self.sounds])
        
    def populate_with_dir(self, dir):
        for file_name in os.listdir(dir):
            parse = file_name.split('.')
            if parse[-1] != 'wav':
                continue
            self.sounds.append(RawSound(self.rate, os.path.join(dir, file_name)))
        return self

class SpreadSound(Sound):
    
    def __init__(self, rate, sound, x_spread, y_spread, t_spread, num_sounds, num_sounds_spread=0):
       self.rate = rate
       self.sound = sound
       self.x_spread = x_spread
       self.y_spread = y_spread
       self.t_spread = t_spread
       self.num_sounds = num_sounds
       self.num_sounds_spread = num_sounds_spread
    
    def render_from(self, location):
        stereo_buffer = np.array([[],[]])
        if self.num_sounds_spread == 0:
            n = self.num_sounds
        else:
            n = max(int(np.random.normal(self.num_sounds, self.num_sounds_spread)), 0)
        reg_index = 0
        for s in range(n):
            center_x, center_y = location
            x = np.random.normal(center_x, self.x_spread) # TODO: Set seeds deterministically so that result for same call is the same
            y = np.random.normal(center_y, self.y_spread)
            sound_reg_pt, sound_data = self.sound.render_from((x, y))
            t = np.random.normal(0, self.t_spread)
            start_index = reg_index + int((t - sound_reg_pt) * self.rate)
            if start_index < 0:
                start_index = abs(start_index)
                stereo_buffer = np.hstack((np.zeros((2, start_index)), stereo_buffer))
                reg_index += start_index
                start_index = 0
            end_index = start_index + sound_data.shape[1]
            if (end_index) >= stereo_buffer.shape[1]:
                stereo_buffer = np.hstack((stereo_buffer, np.zeros((2, end_index - stereo_buffer.shape[1] + 1))))
            stereo_buffer[:, start_index : end_index] = Track._mix(stereo_buffer[:, start_index : end_index], sound_data)
        return (float(reg_index)/self.rate, stereo_buffer)

class ResampledSound(Sound):
    
    def __init__(self, rate, sound, freq_func):
        self.rate = rate
        self.sound = sound
        self.freq_func = np.vectorize(freq_func)
    
    def render_from(self, location):
        block_size = 10000
        reg_pt, sound_data = self.sound.render_from(location)
        index_counter = np.arange(sound_data.shape[1])
        resampled_data = np.array([[],[]])
        end_marker = 2 # outside range of audio data
        # negative times
        i = -1
        stop = False
        mark = reg_pt * self.rate
        while not stop:
            interp_offsets = self._points_for_block(block_size, i)
            interp_points = interp_offsets - interp_offsets[-1] + mark # i_off[-1] is the maximum
            block_left = np.interp(interp_points, index_counter, sound_data[0], left=end_marker)
            block_right = np.interp(interp_points, index_counter, sound_data[1], left=end_marker)
            if block_left[0] == end_marker:
                block_left = block_left[block_left != end_marker]
                block_right = block_right[-block_left.size:]
                stop = True
            block = np.vstack((block_left, block_right))
            resampled_data = np.hstack((block, resampled_data))
            mark = interp_points.min()
            i -= 1
        
        new_reg_pt = float(resampled_data.shape[1])/self.rate
        
        # positive times
        i = 0
        stop = False
        mark = reg_pt * self.rate
        while not stop:
            interp_points = self._points_for_block(block_size, i) + mark
            block_left = np.interp(interp_points, index_counter, sound_data[0], right=end_marker)
            block_right = np.interp(interp_points, index_counter, sound_data[1], right=end_marker)
            if block_left[-1] == end_marker:
                block_left = block_left[block_left != end_marker]
                block_right = block_right[:block_left.size]
                stop = True
            block = np.vstack((block_left, block_right))
            resampled_data = np.hstack((resampled_data, block))
            mark = interp_points.max()
            i += 1
        return (new_reg_pt, resampled_data)
        
    def _points_for_block(self, block_size, block_number):
        block_start_index = block_size * block_number
        block_end_index = block_start_index + block_size
        intervals = self.freq_func(np.arange(block_start_index, block_end_index).astype(np.float)/self.rate)
        resample_points = np.cumsum(intervals)
        return resample_points

class PitchedSound(Sound):
    
    frequency = None
    
    def for_pitch(self, pitch):
        frequency = PitchedSound.resolve_pitch(pitch)
        ratio = frequency/self.frequency
        return ResampledSound(self.rate, self, lambda x, r=ratio: r)
    
    @staticmethod
    def resolve_pitch(pitch):
        try:
            return float(pitch)
        except:
            note_name, octave_str = pitch.split('_')
            return Sound.note_frequency(note_name, int(octave_str))

class RawPitchedSound(RawSound, PitchedSound):
    
    def __init__(self, rate, file_path, registration_point=None, pitch=None):
        parse = os.path.basename(file_path).split('.')
        if registration_point is not None:
            self.reg_pt = registration_point
        else:
            try:
                self.reg_pt = float(parse[1])/1000 # field after first dot in file name is reg_pt ms
            except:
                self.reg_pt = 0.0
        if pitch is None:
            pitch = parse[2] # field after second dot is frequency, Hz
        self.frequency = PitchedSound.resolve_pitch(pitch)
        self.mono_data = self._read_mono_data(rate, file_path)
        self.rate = rate

class RandomPitchedSound(RandomSound, PitchedSound):

    def __init__(self, rate, pitch, sounds=[]):
        self.sounds = sounds
        self.rate = rate
        self.pitch = pitch

    def populate_with_dir(self, dir):
        for file_name in os.listdir(dir):
            parse = file_name.split('.')
            if parse[-1] != 'wav':
                continue
            self.sounds.append(RawPitchedSound(self.rate, os.path.join(dir, file_name)).for_pitch(self.pitch))
        return self

class Instrument:
    
    def __init__(self, data_dir, filter=[]):
        self.note_dict = {}
        for note_file_name in os.listdir(data_dir):
            parse = note_file_name.split('.')
            key = parse[0]
            if not filter != [] and key not in filter:
                continue
            if len(parse) > 2:
                reg_pt = float(parse[1])
            else:
                reg_pt = 0.0
            new_entry = (reg_pt, data_dir + '/' + note_file_name)
            try:
                self.note_dict[key] = self.note_dict[key] + [new_entry]
            except KeyError:
                self.note_dict[key] = [new_entry]
            self.notelist.append(new_entry)
    
    def gen_note_file(self, key=None):
        if key is None:
            notelist = self.notelist
        else:
            notelist = self.note_dict[key]
        return notelist[numpy.random.randint(len(notelist))]
    
    def __getitem__(self, filter):
        return Instrument(self.data_dir, filter)
    
class Track:
    
    def __init__(self, rate, duration, start_time=0, volume=1.0):
        self.rate = rate
        self.top_beat = Beat(parent=self)
        self.start_time = start_time
        self._duration = duration
        self.volume = volume
        self.data = np.zeros((2, int(rate * duration) + 1))
            
    def mix_into(self, t0, buffer):
        self._render()  
        start_index = int(t0 - self.start_time * self.rate) # TODO add support for start time after t0
        data_to_mix = self.data[:, start_index : start_index + buffer.shape[1]]
        return Track._mix(data_to_mix * self.volume, buffer)
    
    @staticmethod
    def _mix(a, b):
        return a + b - a * b
    
    def duration(self):
        return self._duration
    
    def time(self):
        return 0.0
    
    def _render(self): # TODO add parameters t0 and t1, only render inside, and cache results
        for beat in self.top_beat.descendent_beats():
            for sound, location in beat.sounds:
                reg_pt, sound_data = sound.render_from(location)
                start_time = beat.time() - reg_pt
                if start_time < 0:
                    start_index = 0
                    sound_data = sound_data[:, int(start_time * self.rate) :]
                else:
                    start_index = int(start_time * self.rate)
                end_index = start_index + sound_data.shape[1]
                if end_index > self.data.shape[1]:
                    end_index = self.data.shape[1]
                    print (start_index, end_index)
                    sound_data = sound_data[:, : end_index - start_index]
                self.data = np.hstack((self.data[:, :start_index], Track._mix(self.data[:, start_index : end_index], sound_data), self.data[:, end_index:]))
        
        

class Mixer:
    
    def __init__(self, name, rate, tracks=[]):
        self.name = name
        self.tracks = tracks
        self.rate = rate
    
    def play(self, t0=0, t1=None):
        self.render_to_file('temp.wav', t0, t1)
        print "Begin playback."
        winsound.PlaySound('temp.wav', winsound.SND_ALIAS)
        
    def render_to_file(self, out_file_name, t0=0, t1=None, ):
        if not t1:
            t1 = max([track.start_time + track.duration() for track in self.tracks])
        play_buffer = np.zeros((2, int(self.rate * (t1 - t0)) + 1))
        for track in self.tracks:
            play_buffer = track.mix_into(t0, play_buffer)
        play_buffer = (play_buffer * 2**15).astype(np.int16)
        print "Finished rendering, writing out buffer..."
        temp_output = wave.open(out_file_name, 'w')
        temp_output.setparams((2, 2, self.rate, 0, 'NONE', 'not compressed')) # (nchannels, samplewidth, framerate, nframes, compressiontype, compressionname)
        for left_sample, right_sample in zip(play_buffer[0], play_buffer[1]):
            left_bytes = struct.pack('h', left_sample)
            right_bytes = struct.pack('h', right_sample)
            temp_output.writeframes(''.join((left_bytes, right_bytes)))
        temp_output.close()

if __name__ == "__main__":
    print "TEST: Time to resample."
    rate = 44100
    track1 = Track(rate, 5)
    beats = track1.top_beat.split_even(32)
    hit = RandomPitchedSound(rate, "B_3")
    xylo_dir = "audio\\TheLight\\G_major_xylo"
    hit.populate_with_dir(xylo_dir)
    for i, eighth in enumerate(beats):
        eighth.attach(hit, (0, 1))
    testWork = Mixer("Da Bomba Resample", rate, [track1])
    testWork.play(0.0, 4.9)
    