import sys, wave, struct
import numpy as np
import scipy.io.wavfile as wavfile

# Basically just add phase jitter to the whole spectrum, and see whether it makes a convincing continuous sound
# Basically it doesn't, except under special circumstances like nice looping

def write_to_file(rate, out_file_name, buffer):
        data_buffer = (buffer * 2**15).astype(np.int16)
        print np.shape(data_buffer[1])
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
        
filerate, data = wavfile.read(sys.argv[1])
source_sound = np.transpose(np.array(data).astype(np.float) / 2**15)[:,10000:-10000]
transform = np.fft.rfft(source_sound)
N = np.shape(transform)[1]
magnitude = np.abs(transform) * np.array([[1 if i - N < 1000 else 0 for i in range(N)]]*2)
phase_jittered =  magnitude * 5 * np.exp(1j * (np.angle(transform) + 1j * 2 * np.pi * np.random.rand(*np.shape(magnitude))*1))
re_transform = np.fft.irfft(phase_jittered)
print int(44100*4.0/np.shape(re_transform)[1])

write_to_file(44100, "test_tone_out.wav", np.tile(re_transform, int(44100*6.0/np.shape(re_transform)[1])))