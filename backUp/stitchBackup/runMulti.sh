LEFTVIDEOPATH=$shift/storage/190115_apollone_5c3df81f654c320ce82efbd1/01_left.mp4
RIGHTVIDEOPATH=$shift/storage/190115_apollone_5c3df81f654c320ce82efbd1/01_right.mp4

mkdir -p log

T_SH=$(((0*60 + 0)*30 + 0))
L_SH=$(((0*60 + 0)*30 + 0))
R_SH=$(((0*60 + 0)*30 + 43))
HAUT=40
BAS=40

RECTIFY_MAP=0
MATCH_CONF=80
CSS1=160
#CSS1=085
OFFSET_VERIFPANO=$(((3*60 + 0)*30 + 0 - $T_SH))

for param in $@
	do
		sudo rm -rf output/$param.mp4 log/$param.log matches/$param verifPano/$param
		echo "Running $param to construct output/$param.mp4 [...]\nYou can check log/$param.log, matches/$param/*.jpg and verifPano/$param/*.jpg from now."
		sudo nohup ./$param --estimate_infos\
			--v1 $LEFTVIDEOPATH\
			--v2 $RIGHTVIDEOPATH\
			-g $L_SH\
			-d $R_SH\
			-t $T_SH\
			-h $HAUT\
			-b $BAS\
			-r $RECTIFY_MAP\
			-M $MATCH_CONF\
			-C $CSS1\
			-V $OFFSET_VERIFPANO\
			./output/$param.mp4 > log/$param.log 2>&1 &
		sleep 1
	done



# THIS IS A LOG OF MY FIRST SUCCESSFULL STITCH W/O RECTIFY_MAP 
#OPENCV version 3.4
#left shift : 160
#right_shift: 150
#total_shift: 5250
#haut       : 60
#bas        : 60
#rectify_map: 0
#match_conf: 0.6
#css1: 13
#offset_verifPano: 7800
#Finding stitching informations from frame#133111 idx 0
#	stitching infos score: 0
#Finding stitching informations from frame#58351 idx 1
#	stitching infos score: 3.66738
#Finding stitching informations from frame#88578 idx 2
#	stitching infos score: 0
#Finding stitching informations from frame#12365 idx 3
#	stitching infos score: 0
#Finding stitching informations from frame#157161 idx 4
#	stitching infos score: 0
#Finding stitching informations from frame#159510 idx 5
#	stitching infos score: 0
#Finding stitching informations from frame#73036 idx 6
#	stitching infos score: 0
#Finding stitching informations from frame#17661 idx 7
#	stitching infos score: 0
#Finding stitching informations from frame#27776 idx 8
#	stitching infos score: 0
#Finding stitching informations from frame#33642 idx 9
#	stitching infos score: 0
#Finding stitching informations from frame#138294 idx 10
#	stitching infos score: 0
#Finding stitching informations from frame#93007 idx 11
#	stitching infos score: 7.37455
#Finding stitching informations from frame#164454 idx 12
#	stitching infos score: 14.235
#Finding stitching informations from frame#81162 idx 13
#	stitching infos score: 0
#Finding stitching informations from frame#107602 idx 14
#	stitching infos score: 0
#Finding stitching informations from frame#129813 idx 15
#	stitching infos score: 0
#Finding stitching informations from frame#90355 idx 16
#	stitching infos score: 19.7024
#Finding stitching informations from frame#155896 idx 17
#	stitching infos score: 0
#Finding stitching informations from frame#156512 idx 18
#	stitching infos score: 0
#Finding stitching informations from frame#133243 idx 19
#	stitching infos score: 0
#Finding stitching informations from frame#20918 idx 20
#	stitching infos score: 0
#Finding stitching informations from frame#171921 idx 21
#	stitching infos score: 0
#Finding stitching informations from frame#150284 idx 22
#	stitching infos score: 0
#Finding stitching informations from frame#155438 idx 23
#	stitching infos score: 0
#Finding stitching informations from frame#60043 idx 24
#	stitching infos score: 0
#Finding stitching informations from frame#22812 idx 25
#	stitching infos score: 0
#Finding stitching informations from frame#39024 idx 26
#	stitching infos score: 0
#Finding stitching informations from frame#66821 idx 27
#	stitching infos score: 0
#Finding stitching informations from frame#91089 idx 28
#	stitching infos score: 0
#Finding stitching informations from frame#57330 idx 29
#	stitching infos score: 0
#Finding stitching informations from frame#30432 idx 30
#	stitching infos score: 0
#Finding stitching informations from frame#88678 idx 31
#	stitching infos score: 0
#Finding stitching informations from frame#145860 idx 32
#	stitching infos score: 0
#Finding stitching informations from frame#98006 idx 33
#	stitching infos score: 0
#Finding stitching informations from frame#38325 idx 34
#	stitching infos score: 0
#Finding stitching informations from frame#58512 idx 35
#	stitching infos score: 0
#Finding stitching informations from frame#27362 idx 36
#	stitching infos score: 0
#Finding stitching informations from frame#150174 idx 37
#	stitching infos score: 28.9194
#Finding stitching informations from frame#41888 idx 38
#	stitching infos score: 40.3116
#Finding stitching informations from frame#4500 idx 39
#	stitching infos score: 0
#Finding stitching informations from frame#139725 idx 40
#	stitching infos score: 0
#Finding stitching informations from frame#130393 idx 41
#	stitching infos score: 0
#Finding stitching informations from frame#157548 idx 42
#	stitching infos score: 0
#Finding stitching informations from frame#106013 idx 43
#	stitching infos score: 57.4088
#Finding stitching informations from frame#35847 idx 44
#	stitching infos score: 0
#Finding stitching informations from frame#52268 idx 45
#	stitching infos score: 0
#Finding stitching informations from frame#138989 idx 46
#	stitching infos score: 0
#Finding stitching informations from frame#31259 idx 47
#	stitching infos score: 0
#Finding stitching informations from frame#164704 idx 48
#	stitching infos score: 0
#Finding stitching informations from frame#130235 idx 49
#	stitching infos score: 0
#Finding stitching informations from frame#89358 idx 50
#	stitching infos score: 0
#Finding stitching informations from frame#167540 idx 51
#	stitching infos score: 0
#Finding stitching informations from frame#108182 idx 52
#	stitching infos score: 0
#Finding stitching informations from frame#33204 idx 53
#	stitching infos score: 0
#Finding stitching informations from frame#34252 idx 54
#	stitching infos score: 0
#Finding stitching informations from frame#19282 idx 55
#	stitching infos score: 0
#Finding stitching informations from frame#143214 idx 56
#	stitching infos score: 0
#Finding stitching informations from frame#29139 idx 57
#	stitching infos score: 0
#Finding stitching informations from frame#136026 idx 58
#	stitching infos score: 71.0749
#Finding stitching informations from frame#161941 idx 59
#	stitching infos score: 0
#Best feature matching score found: 71.0749
#Find masks from best camera infos found
#Seam boudary is free like a bird
#Seam boudary is free like a bird
#[1984 x 678]warped sizes
#FPS : 29.97
#FPS2 : 29.97
#OpenCV: FFMPEG: tag 0x31637661/'avc1' is not supported with codec id 28 and format 'mp4 / MP4 (MPEG-4 Part 14)'
#OpenCV: FFMPEG: fallback to use tag 0x00000021/'!???'
#Processing frames 0 to 10 over 191953
#Seam boudary is free like a bird
#Seam boudary is free like a bird
#Processing frames 10 to 20 over 191953
