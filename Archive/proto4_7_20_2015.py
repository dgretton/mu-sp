import wave, winsound, struct, os, math, contextlib
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
        total = float(sum(portions))
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
        if sound is None:
            return #  Allow "None" to mean a skip in iterators
        self.sounds += [(sound, location)]
    
    def descendent_beats(self):
        if not self.beats:
            return [self]
        return [self] + list(chain(*[b.descendent_beats() for b in self.beats]))
        
class Sound:
    
    ear_separation = .2
    standard_distance = .2 # the distance at which a sound is imagined to be heard when it's at unit volume
    c_sound = 344.0
    quick_play = False
    
    def render_from(self, location):
        pass
    
    def duration(self):
        pass
            
    def _to_stereo(self, rate, mono_data, location):
        x, y = location
        left_dist = np.sqrt((x + Sound.ear_separation / 2)**2 + y*y)
        right_dist = np.sqrt((x - Sound.ear_separation / 2)**2 + y*y)
        decays = np.array([[Sound.standard_distance/left_dist], [Sound.standard_distance/right_dist]])
        delays = np.array([[left_dist / Sound.c_sound], [right_dist / Sound.c_sound]])
        if Sound.quick_play:
            quick_data = np.hstack((np.zeros((int(delays.max() * rate) + 1,)), mono_data))
            return np.vstack((quick_data, quick_data)) * decays
        padded_data = np.hstack((mono_data, np.zeros((int(delays.max() * rate) + 1,))))
        # A shift of n is realized by a multiplication by exp(2pi*n*w/T) (but it can be fractional!)
        transform = np.tile(np.fft.rfft(padded_data), (2, 1))
        exp_coeff = -2j * np.pi * rate / len(padded_data)
        transformed = transform * np.exp(exp_coeff * delays * np.tile(np.arange(transform.shape[1]), (2, 1)))
        return np.fft.irfft(transformed) * decays
    
    def _read_mono_data(self, rate, filename):
        filerate, data = wavfile.read(filename)
        if filerate != rate:
            print "GOSH DARN IT the file "+filename+" has the wrong rate!"
        return np.array(data).astype(np.float) / 2**15
    
    @staticmethod
    def sigmoid(samples):
        curve = lambda x: 1.0/(1.0 + np.exp(-x))
        return curve(np.arange(-samples/2, samples/2).astype(np.float)/samples * 2 * 5) #  5 "time constants" is nearly 1.0

class RawSound(Sound):
    
    def __init__(self, rate, file_path, registration_point=None, data_cache={}):
        if registration_point is not None:
            self.reg_pt = registration_point
        else:
            parse = os.path.basename(file_path).split('.')
            if len(parse) > 2:
                self.reg_pt = float(parse[1])/1000 # field after first dot in file name is reg_pt ms
            else:
                self.reg_pt = 0.0
        self.rate = rate
        self.file_path = file_path
        self._duration = None
    
    def render_from(self, location):
        mono_data = self._get_from_cache()
        return (self.reg_pt, self._to_stereo(self.rate, mono_data, location))
    
    def duration(self):
        if self._duration is None:
            with contextlib.closing(wave.open(self.file_path,'r')) as f:
                frames = f.getnframes()
                self._duration = frames / float(self.rate)
        return self._duration
    
    def _get_from_cache(self, data_cache={}):
        hits_before_cache = 2
        try:
            hits = data_cache[self.file_path]
            if hits == hits_before_cache:
                mono_data = self._read_mono_data(self.rate, self.file_path)
                data_cache[self.file_path] = mono_data
                return mono_data
            data_cache[self.file_path] = hits + 1
        except ValueError:
            return data_cache[self.file_path]
        except KeyError:
            data_cache[self.file_path] = 1
        return self._read_mono_data(self.rate, self.file_path)

class RandomSound(Sound):
    
    def __init__(self, rate, sounds=None):
        if sounds is None:
            self.sounds = []
        else:
            self.sounds = sounds
        self.rate = rate
        self._duration = None
    
    def render_from(self, location):
        random_sound = self.sounds[np.random.randint(len(self.sounds))]
        return random_sound.render_from(location)
        
    def duration(self):
        if self._duration is None:
            self._duration = max([snd.duration() for snd in self.sounds])
        return self._duration
        
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
    
    def duration(self):
        return self.sound.duration() + self.t_spread * 3  # After 3 standard deviations, probability is acceptably low
    
    def render_from(self, location):
        stereo_buffer = np.array([[],[]])
        if self.num_sounds_spread == 0:
            n = self.num_sounds
        else:
            n = max(int(np.random.normal(self.num_sounds, self.num_sounds_spread)), 0)
        reg_index = 0
        for s in range(n):
            center_x, center_y = location
            if self.x_spread == 0:
                x = center_x
            else:
                x = np.random.normal(center_x, self.x_spread) # TODO: Set seeds deterministically so that result for same call is the same
            if self.y_spread == 0:
                y = center_y
            else:
                y = np.random.normal(center_y, self.y_spread)
            sound_reg_pt, sound_data = self.sound.render_from((x, y))
            if self.t_spread == 0:
                t = 0
            else:
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

class ClippedSound(Sound):
    
    def __init__(self, rate, sound, clip_duration, offset=0.0, margin=.01):
        self.rate = rate
        self.sound = sound
        self.clip_duration = clip_duration
        self.margin = margin
        self.cap = Sound.sigmoid(int(margin * self.rate))
        if offset < 0 or offset > clip_duration:
            print "EEEW NEEW You can't initialize a Clipped Sound that might put the registration point off the sample."
            return
        self.offset = offset
    
    def duration(self):
        return self.clip_duration + self.margin
            
    def render_from(self, location):
        reg_pt, sound_data = self.sound.render_from(location)
        cap_offset = len(self.cap)/2
        start_index = int((reg_pt - self.offset) * self.rate - cap_offset)
        end_index = start_index + int(self.clip_duration * self.rate) + cap_offset
        if start_index < 0:
            start_index = 0
            new_reg_pt = reg_pt
        else:
            new_reg_pt = self.offset
        if end_index > sound_data.shape[1]:
            end_index = sound_data.shape[1]
        clipped_data = sound_data[:, start_index : end_index]
        cap = self.cap[: clipped_data.shape[1]]
        clipped_data[:, : len(cap)] = clipped_data[:, : len(cap)] * cap
        clipped_data[:, -len(cap) :] = clipped_data[:, -len(cap) :] * cap[::-1]
        return new_reg_pt, clipped_data
        
        
class RandomIntervalSound(Sound):
    
    def __init__(self, rate, sound, interval=None, margin=.1, data=None):
        self.rate = rate
        self.sound = sound
        self.interval = interval
        self.margin = margin
        self.data = data
        self.cap = Sound.sigmoid(int(margin * self.rate))
    
    def duration(self):
        return self.interval
    
    def render_from(self, location):
        if self.data is None:
            self.data = self.sound.render_from((0, Sound.standard_distance))[1][0] #  ignore reg pt; take one track; remember it
        total_samples = len(self.data)
        samples = int(self.interval * self.rate)
        unclaimed_samples = samples
        supplementary_intervals = []
        cap_size = len(self.cap)
        while unclaimed_samples > total_samples/2:
            interval_size = np.random.randint(cap_size * 2, total_samples/2)
            unclaimed_samples -= (interval_size - cap_size)
            supplementary_intervals.append(self.random_data_of_length(interval_size))
        mono_data = self.random_data_of_length(unclaimed_samples)
        for subinterval in supplementary_intervals:
            mono_data = np.concatenate((mono_data[: -cap_size], mono_data[-cap_size :] + subinterval[: cap_size], subinterval[cap_size :]))
        return (0.0, self._to_stereo(self.rate, mono_data, location))
    
    def random_data_of_length(self, length):
        random_position = np.random.randint(len(self.data) - length)
        interval_data = np.array(self.data[random_position : random_position + length])
        eff_cap_size = min(len(self.cap), length)
        cap = self.cap[: eff_cap_size]
        interval_data[: eff_cap_size] = cap * interval_data[: eff_cap_size]
        interval_data[-eff_cap_size :] = cap[::-1] * interval_data[-eff_cap_size :]
        return interval_data
    
    def for_interval(self, interval):
        return RandomIntervalSound(rate, self.sound, interval, self.margin, self.data)

class ResampledSound(Sound):
    
    def __init__(self, rate, sound, freq_func):
        self.rate = rate
        self.sound = sound
        self.freq_func = np.vectorize(freq_func)
        self._duration = None
    
    def duration(self):
        if self._duration is None:
            sound_duration = self.sound.duration()
            self._duration = np.trapz(1.0/self.freq_func(np.arange(-sound_duration*self.rate, sound_duration*self.rate)))/self.rate
        return self._duration
    
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
            interp_offsets, next_pt = self._points_for_block(block_size, i)
            interp_points = interp_offsets - next_pt + mark # i_off[-1] is the maximum
            block_left = np.interp(interp_points, index_counter, sound_data[0], left=end_marker)
            block_right = np.interp(interp_points, index_counter, sound_data[1], left=end_marker)
            if block_left[0] == end_marker:
                block_left = block_left[block_left != end_marker]
                if len(block_left) > 0:
                    block_right = block_right[-block_left.size:]
                else:
                    block_right = np.array([])
                stop = True
            block = np.vstack((block_left, block_right))
            resampled_data = np.hstack((block, resampled_data))
            mark = interp_points[0]
            i -= 1
        
        new_reg_pt = float(resampled_data.shape[1])/self.rate
        
        # positive times
        i = 0
        stop = False
        mark = reg_pt * self.rate
        while not stop:
            interp_offsets, next_pt = self._points_for_block(block_size, i)
            interp_points = interp_offsets + mark
            block_left = np.interp(interp_points, index_counter, sound_data[0], right=end_marker)
            block_right = np.interp(interp_points, index_counter, sound_data[1], right=end_marker)
            if block_left[-1] == end_marker:
                block_left = block_left[block_left != end_marker]
                block_right = block_right[:block_left.size]
                stop = True
            block = np.vstack((block_left, block_right))
            resampled_data = np.hstack((resampled_data, block))
            mark = interp_points[0] + next_pt
            i += 1
        return (new_reg_pt, resampled_data)
        
    def _points_for_block(self, block_size, block_number):
        block_start_index = block_size * block_number
        block_end_index = block_start_index + block_size
        intervals = self.freq_func(np.arange(block_start_index, block_end_index).astype(np.float)/self.rate)
        summed = np.cumsum(intervals)
        resample_points = np.hstack(([0], summed[:-1]))
        next_point = summed[-1]
        return (resample_points, next_point)

class PitchedSound(Sound):

    chromatic_scale = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]
    note_map = {name: index for index, name in enumerate(chromatic_scale)}
    note_map.update([("Bb", 1), ("Db", 4), ("Eb", 6), ("Gb", 9), ("Ab", 11)])
    temper_ratio = 2.0**(1.0/12)
    
    def for_pitch(self, pitch):
        frequency = PitchedSound.resolve_pitch(pitch)
        ratio = frequency/PitchedSound.resolve_pitch(self.pitch)
        if abs(ratio - 1.0) < .00001:
            print "Aw yeah, skipped identity repitching."
            return self
        return ResampledSound(self.rate, self, (lambda x, r=ratio: r))
    
    @staticmethod
    def resolve_pitch(pitch):
        try:
            return float(pitch)
        except:
            note_name, octave_str = pitch.split('_')
            return PitchedSound.note_frequency(note_name, int(octave_str))
    
    @staticmethod
    def note_frequency(name, octave=4):
        scale_index = PitchedSound.note_map[name]
        return 440 * PitchedSound.temper_ratio**scale_index * 2**(octave - 4 - (scale_index % 12 > 2))

class RawPitchedSound(RawSound, PitchedSound):
    
    #  TODO: make sure the resampled sound is cached.
    
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
        self.pitch = pitch
        self.rate = rate
        self.file_path = file_path
        self._duration = None

class RandomPitchedSound(RandomSound, PitchedSound):

    def __init__(self, rate, pitch=None, pitched_sounds=None):
        if pitched_sounds is None:
            self.pitched_sounds = []
        else:
            self.pitched_sounds = pitched_sounds
        self.rate = rate
        self.pitch = pitch #  if it's None, it's expected that we'll call for_pitch later
        if pitch is None:
            self.sounds = []
        else:
            self.sounds = [pitched_sound.for_pitch(self.pitch) for pitched_sound in self.pitched_sounds]
        self._duration = None

    def populate_with_dir(self, dir):
        for file_name in os.listdir(dir):
            parse = file_name.split('.')
            if parse[-1] != 'wav':
                continue
            new_pitched_sound = RawPitchedSound(self.rate, os.path.join(dir, file_name))
            self.pitched_sounds.append(new_pitched_sound)
            if self.pitch is not None:
                self.sounds.append(new_pitched_sound.for_pitch(self.pitch))
        return self
    
    def for_pitch(self, pitch): #  override with something smarter than just resampling
        return RandomPitchedSound(self.rate, pitch, self.pitched_sounds)

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
    
    def __init__(self, name, rate, duration, start_time=0, volume=1.0, padding=.1, end_padding=None):
        self.name = name
        self.rate = rate
        self.top_beat = Beat(parent=self)
        self.start_time = start_time
        self._duration = duration
        self.volume = volume
        self.reg_index = 0
        self.pre_padding_samples = padding * self.rate
        if end_padding is None:
            self.post_padding_samples = self.pre_padding_samples
        else:
            self.post_padding_samples = end_padding * self.rate
        self.total_samples = self.pre_padding_samples + int(rate * duration) + self.post_padding_samples + 1
        self.data = np.zeros((2, self.total_samples))
            
    def mix_into(self, t0, buffer): # t0 is mixer time, not track time.
        track_start_index = int((t0 - self.start_time) * self.rate) + self.pre_padding_samples
        buffer_length = buffer.shape[1]
        if track_start_index < 0:
            buffer_write_index = -track_start_index
            if buffer_write_index > buffer_length:
                return buffer
            track_start_index = 0
            print "Be warned, ya dope! You outstayed your welcome in the pre-padding for the " + self.name + " track" 
        else:
            buffer_write_index = 0
        track_end_index = track_start_index - buffer_write_index + buffer_length
        buffer_end_write_index = buffer_length
        if track_end_index > self.total_samples:
            buffer_end_write_index -= track_end_index - self.total_samples
            if buffer_end_write_index < 0:
                return buffer
            track_end_index = self.total_samples
            print "Wowzas warning mah brudda, you've overtaxed track " + self.name + "'s post-padding"
        buffer_data = buffer[:, buffer_write_index : buffer_end_write_index]
        self._render(track_start_index, track_end_index)
        track_data = self.data[:, track_start_index : track_end_index] * self.volume
        buffer[:, buffer_write_index : buffer_end_write_index] = Track._mix(track_data, buffer_data)
        return buffer
    
    @staticmethod
    def _mix(a, b):
        return a + b
    
    def duration(self):
        return self._duration
    
    def time(self):
        return 0.0 #  All times within a track are relative.
    
    def _render(self, track_start_index, track_end_index):
        t0 = (track_start_index - self.pre_padding_samples)/self.rate
        t1 = (track_end_index - self.pre_padding_samples)/self.rate
        sounds_rendered = 0
        for beat in self.top_beat.descendent_beats():
            for sound, location in beat.sounds:
                if not (beat.time() + sound.duration() > t0 and beat.time() - sound.duration() < t1):
                    continue
                sounds_rendered += 1
                reg_pt, sound_data = sound.render_from(location)
                start_time = beat.time() - reg_pt
                start_index = int(start_time * self.rate) + self.pre_padding_samples
                if start_index < 0:
                    start_index = 0
                    sound_data = sound_data[:, -start_index:]
                end_index = start_index + sound_data.shape[1]
                if end_index > self.total_samples:
                    end_index = self.total_samples
                    sound_data = sound_data[:, : end_index - start_index]
                self.data[:, start_index : end_index] = Track._mix(self.data[:, start_index : end_index], sound_data)
        print "Rendered " + str(sounds_rendered) + " sounds for track " + self.name
        
        

class Mixer:

    attenuation_boost = 2
    
    def __init__(self, name, rate, tracks=[]):
        self.name = name
        self.tracks = tracks
        self.rate = rate
    
    def play(self, t0=0, t1=None, quick_play=True):
        self.render_to_file('temp.wav', t0, t1, quick_play=quick_play)
        print "Begin playback."
        winsound.PlaySound('temp.wav', winsound.SND_ALIAS)
        
    def play_beat(self, beat, quick_play=True):
        self.play(beat.time(), beat.time() + beat.duration(), quick_play)
    
    @staticmethod
    def play_sound(sound, location, number=1, quick_play=True):
        Sound.quick_play = quick_play
        buffer = np.array([[],[]])
        for n in range(number):
            buffer = np.hstack((buffer, sound.render_from(location)[1]))
        Mixer.write_to_file(sound.rate, "sound_temp.wav", buffer)
        winsound.PlaySound("sound_temp.wav", winsound.SND_ALIAS)
        
    def render_to_file(self, out_file_name, t0=0, t1=None, quick_play=False):
        check_file_free = open(out_file_name, 'w').close()
        Sound.quick_play = quick_play
        if not t1:
            t1 = max([track.start_time + track.duration() for track in self.tracks])
        data_buffer = np.zeros((2, int(self.rate * (t1 - t0)) + 1))
        for track in self.tracks:
            data_buffer = track.mix_into(t0, data_buffer)
        Mixer.write_to_file(self.rate, out_file_name, data_buffer)
    
    @staticmethod
    def write_to_file(rate, out_file_name, buffer):
        data_buffer = (buffer * 2**15 * Mixer.attenuation_boost).astype(np.int16)
        print "Finished rendering, writing out buffer..."
        print "Max level:"
        print float(data_buffer.max())/2**15
        output_wav = wave.open(out_file_name, 'w')
        output_wav.setparams((2, 2, rate, 0, 'NONE', 'not compressed')) # (nchannels, samplewidth, framerate, nframes, compressiontype, compressionname)
        write_chunk_size = 10000
        write_chunk = ''
        for left_sample, right_sample in zip(data_buffer[0], data_buffer[1]):
            left_bytes = struct.pack('h', left_sample)
            right_bytes = struct.pack('h', right_sample)
            write_chunk += ''.join((left_bytes, right_bytes))
            if len(write_chunk) == write_chunk_size:
                output_wav.writeframes(write_chunk)
                write_chunk = ''
        output_wav.writeframes(write_chunk)
        output_wav.close()

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
    