# convert a .wav file for Micro-Professor to binary or prg
# very much a work in progres

# import libs
import sys
import argparse
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt

# plot specgram, if --make_plots
def plot_specgram(signal_data, sampling_frequency,outfilestub):
    # Plot the signal read from wav file
    plt.figure(figsize=(16, 8), dpi=300)

    plt.title('Spectrogram of a wav file for KIM-1')
    plt.specgram(signal_data[1:],Fs=sampling_frequency,scale='dB')
    plt.xlabel('Time')
    plt.ylim(0,5000)
    plt.ylabel('Frequency')
    plt.savefig(outfilestub + '_specgram.png')

# plot the FFT, if --make_plots
def plot_fft(sound, sampling_freq, outfilestub):
    fft_spectrum = np.fft.rfft(sound)
    freq = np.fft.rfftfreq(sound.size, d=1./sampling_freq)
    fft_spectrum_abs = np.abs(fft_spectrum)
    plt.figure(figsize=(16, 5), dpi=300)
    plt.plot(freq[freq<3500], fft_spectrum_abs[freq<3500])
    plt.xlabel("frequency, Hz")
    plt.ylabel("Amplitude, units")
    plt.ylim(0,1.1*fft_spectrum_abs[freq>1000].max())
    plt.xlim(500,3500)
    plt.savefig(outfilestub + '_fft.png')

# output some diagnostic plot of the waveform
def plot_start(starttime,time,sound,outfilestub,title):
    newvec = (time>starttime) & (time<(starttime+0.02))
    plt.figure(figsize=(16, 5), dpi=300)
    plt.plot(time[newvec], sound[newvec])
    plt.title(title)
    plt.savefig(outfilestub + '_start_waves_short.png')

# return a bit based on frequency counts
def returnbit(count1000,count2000):
    mpfbit = 0
    time1000 = count1000/1000
    time2000 = count2000/2000
    # note: had to add this because for hypertape high-freq counts may 
    # be underestimated, this is something that needs further attention!
    #print(count2400,count3700)
    if time2000 > 0 and (time1000/time2000 > 1.5):
        mpfbit = 1
    # this can't be zero; warning nevertheless.
    elif time2000 == 0:
        print("WARNING: potentially mangled bit!\nreturning 0 as default")
    # return single bit
    return mpfbit

# compute the checksum for the code
def compute_chsum(message):
    hexsymbols = '0123456789ABCDEF'
    chsum = 0
    message_bytearray = bytearray()
    #print(message)
    hexnum = -1
    for i in range(0,len(message),2):
        if message[i] not in hexsymbols:
            print("WARNING: not valid hex symbol")
        numchar1 = int('{:08b}'.format(ord(message[i]))[4:],2)
        numchar2 = int('{:08b}'.format(ord(message[i+1]))[4:],2)
        if ord(message[i]) < 0x41:
            numchar1 = '{:04b}'.format(numchar1)
        else:
            numchar1 += 9
            numchar1 = '{:04b}'.format(numchar1)
        if ord(message[i+1]) < 0x41:
            numchar2 = '{:04b}'.format(numchar2)
        else:
            numchar2 += 9
            numchar2 = '{:04b}'.format(numchar2)

        totnumchar = numchar1 + numchar2
        chsum += int(totnumchar,2)
        message_bytearray.append(int(totnumchar,2))

    return '{:04x}'.format(chsum),message_bytearray

# return data vector and sample rate from wav
def return_data_from_wav(input_file):

    wav = wavfile.read(input_file)
    samplerate = wav[0]
    data = wav[1]
    sound = ''
    # assuming mono, if stereo deciding which channel
    # note: there is a potential weakness; if mute channel
    # has still lots of noise it could be inadvertently selected
    # this needs some work to make full-proof. 
    if len(data.shape) == 2:
           maxsignal1 = data[:,0].max()
           maxsignal2 = data[:,1].max()
           if maxsignal1 >= maxsignal2:
               sound = data[:,0]
           else:
               sound = data[:,1]
    elif len(data.shape) == 1:
        sound = data
    maxsignal = sound.max()
    minsignal = sound.min()
    
    if minsignal >= 0:
        sound2 = list()
        for value in sound:
            sound2.append(value - int(0.5*maxsignal)+1)
        sound = np.array(sound2)
        minsignal = sound.min()
        maxsignal = sound.max()
    length = sound.shape[0] / samplerate # aantal seconden
    time = np.linspace(0., length, sound.shape[0])
    return sound,maxsignal,minsignal,time,samplerate

# only if debug, print stuff
def debug_timing(count1000, count2000, mpfbit, cycleup):
    print("time: {:.4f}\ttime1000: {:4f}\ttime2000: {:4f}\tbit: {}".format(cycleup,count1000/(2*1000),count2000/(2*2000),mpfbit))

# only if debug, print stuff
def debug_decoding(bytecounter, newbyte, hexbyte, cycleup):
    print("time: {:.4f}\tbyte {}: {}\tparity {}\tbin: {:08b}\thex: {}\tASCII: {}".format(cycleup, bytecounter, newbyte[:8][::-1], newbyte[7], int(hexbyte,16),hexbyte, chr(int(hexbyte,16))))


# if new bit read, add it
def add_bit(totaltime,bitcounter, bytecounter, newbyte,code_hexlist,cycleup,count1000, count2000):
    mpfbit = returnbit(count1000, count2000)
    if debug:
        debug_timing(count1000,count2000,mpfbit,cycleup)
    totaltime = totaltime + (count1000/(1000)) + (count2000/(2000))
    bitcounter += 1
    newbyte += str(mpfbit)
    hexbyte='0'
    oldbyte='0'
    if bitcounter == 10:
        hexbyte = hex(int(newbyte[1:9][::-1], 2))
        if debug:
            debug_decoding(bytecounter,newbyte,hexbyte,cycleup)
        code_hexlist.append(hexbyte)
        oldbyte = newbyte
        newbyte = ''
        bitcounter = 0
        bytecounter += 1
    return totaltime, bitcounter, bytecounter, newbyte, code_hexlist, oldbyte


####################
# block arguments and parsing
parser = argparse.ArgumentParser( description='Convert WAV to binary PRG for MPF')
parser.add_argument("-i", "--input", help="myrecording.wav", type=str, nargs=1)
parser.add_argument("-o", "--output", help="output PRG", type = str, default='test.prg')
parser.add_argument("-d", "--debug", help="debug", action="store_true")
parser.add_argument("-b", "--as_bin", help="output as BIN", action="store_true")
parser.add_argument("-p", "--as_prg", help="output as PRG", action="store_true")
parser.add_argument("-s", "--make_plots", help="output plots", action="store_true")

args = parser.parse_args()

input_file = args.input[0]
output_prg = args.output
debug = args.debug
as_bin = args.as_bin
as_prg = args.as_prg
make_plots = args.make_plots

if as_prg and as_bin:
    print("WARNING! (fatal): you should specify only one type of output format")
    sys.exit()
# end block args and parsing
####################

# convert wav to soundvector and samplerate
sound,maxsignal,minsignal,time,samplerate = return_data_from_wav(input_file)
if make_plots:
    plot_specgram(sound,samplerate,output_prg.split('.')[0])
    plot_fft(sound, samplerate, output_prg.split('.')[0])

# initialize a bunch of vars

count1000 = 0
count2000 = 0
newbit = 'no'
totaltime = 0
firsttime = 0
lasttime = 0
bitcounter = 0
bytecounter = 0 
newbyte = ''
code_hexlist = list()
allshort = list()
alllong = list()
lastsignal = 0
triggerup = 0
triggerdown = 0
cycledown = 0
cycleup = -1
prevhightime = -1
cycletime = 0
allmessages = list()
leadertone_detected = False
lastgoodpeak = 0

min2000 = 2
min1000 = 1
minfreq2000 = 1600
minfreq1000 = 700
maxfreq1000 = 1300
maxfreq2000 = 2400

# go through every timepoint in sound vector
for i,x in enumerate(sound):
    # for the time being the script uses default trigger levels
    # note: consider making this dynamic, and/or add hysteresis
    triggerlevel = 0.5
    if x > triggerlevel * maxsignal or x < triggerlevel * minsignal:
        
        if x > triggerlevel * maxsignal and triggerup == 0:
            
            cycleup = time[i]
            triggerup = 1
            triggerdown = 0

        if x < triggerlevel * maxsignal and triggerdown == 0:
            cycledown = time[i]
            triggerdown = 1
        
        lastsignal = x

        if triggerup == 1 and triggerdown == 1 and cycleup > 0: 

            if firsttime == 0:
                firsttime = time[i]
            if lasttime < time[i]:
                lasttime = time[i]

            halfcycletime = cycledown - cycleup
            cycletime = cycleup - prevhightime
            prevhightime = cycleup
            triggerup = 0
            triggerdown = 0
            hertz = (1/cycletime)
            if debug:
                print(hertz)
            # if within high-freq signal level, and shift from low freq
            if hertz > minfreq2000 and hertz < maxfreq2000 and newbit == 'yes':
                betweengoodpeaks = lastgoodpeak - cycleup
                lastgoodpeak = cycleup
                # evaluate if sufficient good signal seen to call bit
                if count2000 > min2000 and count1000 > min1000 and betweengoodpeaks < (3*(1/1000)):

                    # add bit
                    totaltime,bitcounter, bytecounter,newbyte,code_hexlist,oldbyte = add_bit(totaltime,bitcounter,bytecounter, newbyte,code_hexlist, cycleup, count1000, count2000)

                    if bitcounter == 0:
                        oldbyte = ''
                    # only if --make_plots
                    if make_plots:
                        # plot the start of the sequence for diagnostic purposes
                        if leadertone_detected:
                            title='At the start of the message'
                            plot_start(cycleup,time,sound,output_prg.split('.')[0]+'{:.4f}'.format(cycleup),title)
                        
                # if not meeting proper criteria, warn for possible problems
                else:
                    if count1000 > 100:
                        print('  finished reading 1000 hz leadertone at {:.2f} s'.format(time[i]))
                        title='At the end of 1000 hz leadertone'
                        plot_start(cycleup,time,sound,output_prg.split('.')[0]+'{:.4f}'.format(cycleup),title)

                    else:
                        print("\nWARNING: possible mangled/skipped bit at time {:.3f} \nor larger gap in message.".format(cycleup))
                        if debug:
                            title = 'Possible mangled/skipped bit'
                            offset = 9*((count1000/(1000)) + (count2000/(2000)))
                            plot_start(cycleup-offset,time,sound,output_prg.split('.')[0]+'{:.4f}'.format(cycleup),title)
                    
                    
                # re-initialize counts    
                count2000 = 1
                count1000 = 0

                # re-initialize variable that keeps track of shifting from high to low freq
                newbit = 'no'
                leadertone_detected = False
                
                # keep track of frequency in the high-pitch class  
                allshort.append(hertz)

            # if high frequency but no shift to low freq detected prior, just countup
            elif hertz > minfreq2000 and hertz < maxfreq2000:
                lastgoodpeak = cycleup
                count2000 += 1
                allshort.append(hertz)
                if count2000 == 100:
                    print("  2000 hz long tone detected at {:.2f} s".format(time[i]))
                    leadertone_detected = True
            # if low frequency, countup, and detect shift to low freq
            elif hertz <= maxfreq1000 and hertz > minfreq1000:
                lastgoodpeak = cycleup
                
                newbit = 'yes'
                count1000 += 1
                if count1000 == 100:
                    print("\n\n  leader 1000 hz tone detected at {:.2f} s".format(time[i]))
                    leadertone_detected = True
                alllong.append(hertz)

        prevhighlowtime = time[i]



print('\n\nHexdump:\n  ', end = '')
#print(code_hexlist)
for i in range(1,len(code_hexlist)+1):
    if i % 20 > 0:
        print(' {:02x}'.format(int(code_hexlist[i-1],16)),end = '')
    else:
        print(' {:02x}'.format(int(code_hexlist[i-1],16)))
        if i < len(code_hexlist):
            print('  ', end = '')
print()       

computed_checksum = '{:4x}'.format(np.array([int(x,16) for x in code_hexlist][7:]).sum())
print('\nComputed checksum: {}'.format(computed_checksum))

output_file = output_prg
codestring = ''.join(['{:02x}'.format(int(byte,16)) for byte in code_hexlist])
mc_bytes = bytearray.fromhex(codestring)
outbytearrayfh = open(output_file,'wb')

if as_bin:
    outbytearrayfh.write(mc_bytes[7:])
elif as_prg:
    outbytearrayfh.write(mc_bytes[2:4])
    outbytearrayfh.write(mc_bytes[7:])
else:
    print("WARNING: no output generated, specify format")
outbytearrayfh.close()

# print out some information
print("\n  first signal: {:.2f}\n  last signal: {:.2f}".format(firsttime, lasttime))

print("  totaltime, estimated from waves: {:.2f}".format(totaltime))

print("  short pulses, estmate of hz: {:.2f}".format(np.mean(np.array(allshort))))
print("  long pulses, estmate of hz: {:.2f}".format(np.mean(np.array(alllong))))
