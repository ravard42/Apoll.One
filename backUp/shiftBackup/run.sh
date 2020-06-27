LEFT_SRC=storage/190115_apollone_5c3df81f654c320ce82efbd1/01_left.mp4
RIGHT_SRC=storage/190115_apollone_5c3df81f654c320ce82efbd1/01_right.mp4

rm -rf shifting
mkdir -p shifting

#T_SH=$(((2*60 + 0)*30 + 0))
#L_SH=$(((0*60 + 14)*30 + 0))
#R_SH=$(((0*60 + 0)*30 + 14))
T_SH=$(((0*60 + 0)*30 + 0))
L_SH=$(((0*60 + 0)*30 + 0))
R_SH=$(((0*60 + 0)*30 + 43))

./$1 --v1 $LEFT_SRC\
		--v2 $RIGHT_SRC\
		-g $L_SH\
		-d $R_SH\
		-t $T_SH\
