#!/bin/sh
# Render every scene at 1080p30 and join them into offmesh-explainer.mp4.
set -e
cd "$(dirname "$0")"
M=${MANIM_ENV:-../menv}/bin
$M/python vidnums.py
SCENES="S0Intro S1OurROM S2Bottleneck S3Idea S4Points S5Twist S6Payoff3D S7Results2D S8Summary"
printf "%s\n" $SCENES | xargs -P 4 -I{} sh -c "$M/manim --resolution 1920,1080 --frame_rate 30 --disable_caching offmesh.py {} > {}.hq.log 2>&1 && echo {} done"
: > concat.txt
for s in $SCENES; do echo "file 'media/videos/offmesh/1080p30/$s.mp4'" >> concat.txt; done
$M/ffmpeg -v error -y -f concat -safe 0 -i concat.txt -c copy offmesh-explainer.mp4
$M/ffprobe -v error -show_entries format=duration,size -of default=nw=1 offmesh-explainer.mp4
