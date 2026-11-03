#######################################
# bench2
#######################################

reset
#set terminal postscript eps enhanced solid color 
#set output "results_cc2d_bench2_reference_u2.eps"
#set terminal png
#set output "results_cc2d_bench2_reference_u2.png"
set xlabel "time"
set ylabel "u2(0.5,0.2)"
set xrange [9.30:9.69]
set yrange [-1:1]

plot './logs/results_cc2d_bench2_referencenew/pointvalues_q2_lv2_dt4' using ($2+0.086252):17 with lines lw 3 lt rgb "red"            title 'u2(0.5,0.2), Lv. 2',\
     './logs/results_cc2d_bench2_referencenew/pointvalues_q2_lv3_dt4' using ($2+0.019063):17 with lines lw 3 lt rgb "blue"           title 'u2(0.5,0.2), Lv. 3',\
     './logs/results_cc2d_bench2_referencenew/pointvalues_q2_lv4_dt4' using ($2+0.012031):17 with lines lw 3 lt rgb "dark-goldenrod" title 'u2(0.5,0.2), Lv. 4',\
     './logs/results_cc2d_bench2_referencenew/pointvalues_q2_lv5_dt4' using ($2+0.002675):17 with lines lw 3 lt rgb "dark-green"     title 'u2(0.5,0.2), Lv. 5',\
     './logs/results_cc2d_bench2_referencenew/pointvalues_q2_lv6_dt4' using 2:17 with lines lw 3 lt rgb "black"          title 'u2(0.5,0.2), Lv. 6'

set terminal win
set output
