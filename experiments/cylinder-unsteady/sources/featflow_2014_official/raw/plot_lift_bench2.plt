#######################################
# bench2
#######################################

reset
#set terminal postscript eps enhanced solid color 
#set output "results_cc2d_bench2_reference_lift.eps"
#set terminal png
#set output "results_cc2d_bench2_reference_lift.png"
set xlabel "time"
set ylabel "C_L"
set xrange [9.30:9.69]
set yrange [-1.05:1.05]

plot './logs/results_cc2d_bench2_referencenew/bdforces_q2_lv2_dt4' using ($2+0.086252):5 with lines lw 3 lt rgb "red"            title 'Lift Lv. 2',\
     './logs/results_cc2d_bench2_referencenew/bdforces_q2_lv3_dt4' using ($2+0.019653):5 with lines lw 3 lt rgb "blue"           title 'Lift Lv. 3',\
     './logs/results_cc2d_bench2_referencenew/bdforces_q2_lv4_dt4' using ($2+0.012181):5 with lines lw 3 lt rgb "dark-goldenrod" title 'Lift Lv. 4',\
     './logs/results_cc2d_bench2_referencenew/bdforces_q2_lv5_dt4' using ($2+0.002675):5 with lines lw 3 lt rgb "dark-green"     title 'Lift Lv. 5',\
     './logs/results_cc2d_bench2_referencenew/bdforces_q2_lv6_dt4' using 2:5 with lines lw 3 lt rgb "black"          title 'Lift Lv. 6'

set terminal win
set output
