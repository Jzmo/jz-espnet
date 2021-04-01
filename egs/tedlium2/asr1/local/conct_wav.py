import subprocess
import sys
import pdb


before = 0
after = 0

utts = {}

# sox AimeeMullins_2009P.wav seg1.wav trim 0017.82 =0028.
wav_file = open(sys.argv[1], 'r')
orig_wavs = wav_file.readlines()

seg_file = open(sys.argv[2], 'r')
orig_segs = seg_file.readlines()

odir = sys.argv[3]

# sph2pipe -f wav - p / home/ubuntu/wtianzi1/work/espnet/egs/tedlium2/asr1/db/TEDLIUM_release2/test/sph/AimeeMullins_2009P.sph AimeeMullins_2009P.wav

for line in orig_wavs:
    splited = line.strip().split(" ")
    uid = splited[0]
    owav = odir + '/' + uid+'.wav'
    cmd = " ".join(splited[1:-1]
                   + [owav])
    f = open(odir + '/' + "sph2wav.sh", "a")
    f.write(cmd + '\n')
    f.close()
code = subprocess.run(['bash', odir + '/' + 'sph2wav.sh'])

# sox AimeeMullins_2009P.wav seg1.wav trim 0017.82 =0028.
# wav2seg.sh
code = subprocess.run(['bash', odir + '/' + 'wav2seg.sh'])
# sox track1.wav silence.wav track2.wav silence.wav ... output.wav
# seg2scp.sh
for line in orig_segs:
    seg, full, start, end = line.strip().split(" ")
    wav_full = odir + '/' + full + '.wav'
    wav_seg = odir + '/' + seg + '.wav'
    cmd = " ".join(['sox', wav_full, wav_seg, 'trim', start, '='+str(end)])
    f = open(odir + '/' + "wav2seg.sh", "a")
    f.write(cmd + '\n')
    f.close()

for line in orig_wavs:
    splited = line.strip().split(" ")
    sid = splited[0]
    cmd = " ".join([sid, "sox -t wav ", odir + '/'+sid+"-*.wav", "-t wav - |"])
    f = open(odir + '/' + "wav.scp", "a")
    f.write(cmd + '\n')
    f.close()
