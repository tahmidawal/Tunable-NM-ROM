#######################################
# bench2
#######################################

reset
#set terminal postscript eps enhanced solid color 
#set output "results_cc2d_bench2_dtcomp_pressure.eps"
#set terminal png
#set output "results_cc2d_bench2_dtcomp_pressure.png"
set xlabel "time"
set ylabel "P1-P2"
set xrange [9.30:9.69]
set yrange [2.415:2.51]

plot './logs/results_cc2d_bench2_referencenew/pointvalues_q2_lv6_dt1' using ($2-0.0327):($7-$12) with lines lw 3 lt rgb "dark-goldenrod" title 'P1-P2, Lv. 6, dt=1/100',\
     './logs/results_cc2d_bench2_referencenew/pointvalues_q2_lv6_dt2' using ($2-0.002):($7-$12) with lines lw 3 lt rgb "blue"     title 'P1-P2, Lv. 6, dt=1/400',\
     './logs/results_cc2d_bench2_referencenew/pointvalues_q2_lv6_dt4' using 2:($7-$12) with lines lw 3 lt rgb "black"          title 'P1-P2, Lv. 6, dt=1/1600'

set terminal win
set output
