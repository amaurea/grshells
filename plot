#!/bin/bash
dt=$(awk -v dti=$3 'BEGIN{printf("%.1f\n",dti*0.1)}')
if [[ $1 =~ 0*([0-9]+)\.txt ]]; then
	ti=${BASH_REMATCH[1]}
	t=$(awk -v ti=$ti 'BEGIN{printf("%.1f\n", ti*0.1-100)}')
	R=$(awk -v ti=$ti 'BEGIN{A=7.923007e-2;t=(ti*0.1-100);print t<0?0:A*t**(1.0/3)}')
fi
echo "
set samples 10000
set xlabel 'r (l_P)'
set title sprintf('t: %8.1f t_P  dt: %4.1f',-$t,$dt)
set yrange [1e-4:1e4]
set xrange [0:3]
set logscale y
set format y '10^{%T}'
set key invert
f(x) = x < 1e-6 ? 1e-6 : x
set term pngcairo size 800,600 enhanced
set output '$2'
plot '$1' u 1:(1/f(1-$R/\$1)) w l title 'Schw b' lt 2 lc rgb '#ffc0ff', '' u 1:(f(1-$R/\$1)) w l title 'Schw a' lt 2 lc rgb '#c0c0ff', '' u 1:5 w l title 'b' lt 1 lc rgb '#ff00ff', '' u 1:4 w l title 'a' lt 1 lc rgb '#0000ff', '' u 1:3 w l title 'R' lt 1 lc rgb '#00ff00', '' u 1:(f(\$2)) w l title 'dR' lt 1 lc rgb '#ff0000'" | gnuplot
