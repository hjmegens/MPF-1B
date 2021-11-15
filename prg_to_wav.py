# general usage:
# if general .bin file provide start address
# python prg_to_wav.py -i VUTAPE.BIN -o vutape.wav -s 0000
# if commodore style .prg file
# python prg_to_wav.py -i first_example_first_book_of_KIM.prg -o first_example_first_book_of_KIM.wav --input_is_prg 

# import libs
import argparse
from scipy.io import wavfile
import numpy as np

# compute the checksum for the code
def compute_chsum(message):
    chsum = sum(message)
    return '{:04x}'.format(chsum)


# add 'bytes' in ascii to bitstring
def add_byte(bitstring,addchars):
    for i in range(0,len(addchars),2):
        revbyte = '{:08b}'.format(int(addchars[i:i+2],16))[::-1]
        #print(revbyte)
        bitstring = bitstring + '0' + revbyte + '1'
        #print(bitstring)
    return bitstring


# convert from byte code to ascii
def code_from_byte_to_ascii(code):
    asciicode = ''
    for byte in code:
        asciichar = '{:02x}'.format(byte)
        asciicode += asciichar.upper()
    return asciicode

def separate_code_and_start_addres(mc_bytes, is_prg, start_address):
    code = 0
    sa = 0
    if is_prg:
        code = mc_bytes[2:]
        sa = '{:04x}'.format(mc_bytes[1]*(16**2) + mc_bytes[0])
        sal = mc_bytes[0]
        sah = mc_bytes[1]
        ea = '{:04x}'.format(int(sa,16)+len(mc_bytes[2:])-1)
    elif (not is_prg) and (start_address != 'NA'):
        code = mc_bytes
        sa = '{:04x}'.format(int(start_address, 16))
        ea = '{:04x}'.format(int(sa,16)+len(mc_bytes[2:])-1)
    else:
        print("WARNING: no start address can be defined")
        code = mc_bytes
        sa = '0300'
        ea = '{:04x}'.format(int(sa,16)+len(mc_bytes[2:])-1)

    return (code, sa, ea)

def convert_id(ID):
    hexsymbols = '0123456789ABCDEF'
    IDH = 0
    IDL = 0
    if len(ID) == 4:
        for character in ID:
            if character not in hexsymbols:
                print("WARNING: ID contains non hex symbols.")
        IDH = ID[:2]
        IDL = ID[2:]

    else:
        print("WARNING! Provided ID is not 4 digits wide!")

    return IDL, IDH

# convert the string of bits to array
def convert_bitstring_to_array(bitstring, onebit, zerobit):
    message_list = list()
    for bit in bitstring:
        if bit == '1':
            message_list += onebit
        if bit == '0':
            message_list += zerobit
    return message_list

####################
# block arguments and parsing
parser = argparse.ArgumentParser( description='Convert binary PRG to WAV for Micro-Professor MPF-1B')
parser.add_argument("-i", "--input_prg_or_binary", help="myprogram.bin or myprogram.prg", type=str, nargs=1)
parser.add_argument("-b", "--bitdepth", help="currently only 16 bits supported",type=int, default=16)
parser.add_argument("-p", "--input_is_prg", help="if true no start address needed", action="store_true")
parser.add_argument("-n", "--program_name_id", help="four-digit program id", type=str, default='01', nargs=1)
parser.add_argument("-s", "--start_address", help="start addres", type=str, default='NA')
parser.add_argument("-o", "--output", help="output WAV", type = str, default='out.wav')

args = parser.parse_args()

input_file = args.input_prg_or_binary[0]
is_prg = args.input_is_prg
bitdepth = args.bitdepth
ID = args.program_name_id[0]
start_address = args.start_address
output_WAV = args.output
# end block args and parsing
####################

#open a binary file; note the code is expected to have
# two leading bytes indicating the address, in the 
# commodore 'PRG' file style
hexstringfh = open(input_file, 'rb')
hexstring = hexstringfh.read().hex()
hexstringfh.close()
mc_bytes = bytearray.fromhex(hexstring)

# generate separate code block and 
# start address
code, SA, EA = separate_code_and_start_addres(mc_bytes, is_prg, start_address)
print("Start address: {}".format(SA))
print("End address: {}".format(EA))
# For now just 16 bits (32767 -32768); 
# maybe other encoding later. 
bits = bitdepth

# note that the only samplerate supported currently is 44100
# every bit, long or short, is 324 points, i.e. 7.35 ms.
oneshort = 11 * [2**(bits-1)-1] +  11 * [-1*(2**(bits-1))]
onelong = 22 * [2**(bits-1)-1] + 22 * [-1*(2**(bits-1))] 
zerobit = 8 * oneshort + 2 * onelong
onebit = 4 * oneshort + 4 * onelong

leader_low = 4000 * onelong
interim_high = 4000 * oneshort
end_high = 8000 * oneshort

# create high and low adress byte
SAL = SA[2:]
SAH = SA[:2]

EAL = EA[2:]
EAH = EA[:2]

asciicode = code_from_byte_to_ascii(code)
print("asciicode: ", asciicode)
#ASCIICODE='A510A6118511861000'

# check if ID is allowed: should be hex and not include 0 or 0xFF
print('ID given is: {}'.format(ID))
IDL, IDH = convert_id(ID)
print("IDH is: {}; IDL is {}".format(IDH,IDL))

# compute checksum; indeed this is a bit convoluted
# could have been done directly on the code: 
# chsumdirect = sum(mc_bytes)
# print('checksum directly from binary: {:04x}'.format(chsumdirect))
# but this script is primarily learning to understand the bits 
# and pieces of the cassette interface.

# compute checksum
chsum = compute_chsum(code)
print('checksum: {}'.format(chsum))

# separate low and high byte
# needs to be converted to uppercase
chsumH = chsum[:2].upper()
chsumL = chsum[2:].upper()
bitstring1 = ''
bitstring1 = add_byte(bitstring1, IDL)
bitstring1 = add_byte(bitstring1, IDH)
# then the address low byte
bitstring1 = add_byte(bitstring1, SAL)
# followed by addres high byte
bitstring1 = add_byte(bitstring1, SAH)
bitstring1 = add_byte(bitstring1, EAL)
bitstring1 = add_byte(bitstring1, EAH)
bitstring1 = add_byte(bitstring1, chsum[2:])
# next the code, in ascii
bitstring2 = ''
bitstring2 = add_byte(bitstring2, asciicode)

# convert the string of bits to array
part1_as_array = convert_bitstring_to_array(bitstring1, onebit, zerobit)
part2_as_array = convert_bitstring_to_array(bitstring2, onebit, zerobit)

# define 4s worth of a no-sound leader and end in the WAV
leaderzero = 44100 * 2 * [0]
endzero = 44100 * 2 * [0]

# string everything together: 4 seconds no sound, followed by 
# the actual message, ended by 4 seconds of no-sound

total_vector = np.array(leaderzero + leader_low + part1_as_array + interim_high + part2_as_array + end_high + endzero, dtype='int'+str(bits))

# report total time of the WAV
print("WAV total time: {:.2f} seconds.".format(len(total_vector)/44100))

# write vector to WAV
wavfile.write(output_WAV, 44100, total_vector)
