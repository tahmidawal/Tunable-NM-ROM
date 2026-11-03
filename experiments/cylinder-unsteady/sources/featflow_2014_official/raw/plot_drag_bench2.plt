#######################################
# bench2
#######################################

reset
#set terminal postscript eps enhanced solid color 
#set output "results_cc2d_bench2_reference_drag.eps"
#set terminal png
#set output "results_cc2d_bench2_reference_drag.png"
set xlabel "time"
set ylabel "C_D"
set xrange [9.30:9.69]
set yrange [3.14:3.23]

plot './logs/results_cc2d_bench2_referencenew/bdforces_q2_lv2_dt4' using ($2+0.086252):4 with lines lw 3 lt rgb "red"            title 'Drag Lv. 2',\
     './logs/results_cc2d_bench2_referencenew/bdforces_q2_lv3_dt4' using ($2+0.019653):4 with lines lw 3 lt rgb "blue"           title 'Drag Lv. 3',\
     './logs/results_cc2d_bench2_referencenew/bdforces_q2_lv4_dt4' using ($2+0.012181):4 with lines lw 3 lt rgb "dark-goldenrod" title 'Drag Lv. 4',\
     './logs/results_cc2d_bench2_referencenew/bdforces_q2_lv5_dt4' using ($2+0.002675):4 with lines lw 3 lt rgb "dark-green"     title 'Drag Lv. 5',\
     './logs/results_cc2d_bench2_referencenew/bdforces_q2_lv6_dt4' using 2:4 with lines lw 3 lt rgb "black"          title 'Drag Lv. 6'

set terminal win
set output
