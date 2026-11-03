#######################################
# bench2
#######################################

reset
#set terminal postscript eps enhanced solid color 
#set output "results_cc2d_bench2_dtcomp_drag.eps"
#set terminal png
#set output "results_cc2d_bench2_dtcomp_drag.png"
set xlabel "time"
set ylabel "C_D"
set xrange [9.30:9.69]
set yrange [3.16:3.23]

plot './logs/results_cc2d_bench2_referencenew/bdforces_q2_lv6_dt1' using ($2-0.0327):4 with lines lw 3 lt rgb "dark-goldenrod" title 'Drag Lv. 4, dt=1/100',\
     './logs/results_cc2d_bench2_referencenew/bdforces_q2_lv6_dt2' using ($2-0.002):4 with lines lw 3 lt rgb "blue"     title 'Drag Lv. 5, dt=1/400',\
     './logs/results_cc2d_bench2_referencenew/bdforces_q2_lv6_dt4' using 2:4 with lines lw 3 lt rgb "black"          title 'Drag Lv. 6, dt=1/1600'

set terminal win
set output
