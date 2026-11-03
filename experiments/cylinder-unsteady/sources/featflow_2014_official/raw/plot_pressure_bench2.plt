#######################################
# bench2
#######################################

reset
#set terminal postscript eps enhanced solid color 
#set output "results_cc2d_bench2_reference_pressure.eps"
#set terminal png
#set output "results_cc2d_bench2_reference_lift.png"
set xlabel "time"
set ylabel "P1-P2"
set xrange [9.30:9.69]
set yrange [2.415:2.52]

plot './logs/results_cc2d_bench2_referencenew/pointvalues_q2_lv2_dt4' using ($2+0.086252):($7-$12) with lines lw 3 lt rgb "red"            title 'P1-P2 Lv. 2',\
     './logs/results_cc2d_bench2_referencenew/pointvalues_q2_lv3_dt4' using ($2+0.019653):($7-$12) with lines lw 3 lt rgb "blue"           title 'P1-P2 Lv. 3',\
     './logs/results_cc2d_bench2_referencenew/pointvalues_q2_lv4_dt4' using ($2+0.012181):($7-$12) with lines lw 3 lt rgb "dark-goldenrod" title 'P1-P2 Lv. 4',\
     './logs/results_cc2d_bench2_referencenew/pointvalues_q2_lv5_dt4' using ($2+0.002675):($7-$12) with lines lw 3 lt rgb "dark-green"     title 'P1-P2 Lv. 5',\
     './logs/results_cc2d_bench2_referencenew/pointvalues_q2_lv6_dt4' using 2:($7-$12) with lines lw 3 lt rgb "black"          title 'P2-P2 Lv. 6'

set terminal win
set output
