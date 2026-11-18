import sys,subprocess,argparse
ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();subprocess.run([sys.executable,'c3_global_compress.py','--out',a.out,'--ranks','384,512','--orders','64,96,128,192'],check=True)
