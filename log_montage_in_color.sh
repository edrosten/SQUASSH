#!/bin/bash

if [[ $# != 1 && $# != 1 ]]
then
	cat<<-FOOD
	Incorrect args 
	$0 <log_dir> 
	FOOD
	exit 1
fi

cols=10

innerdir="$1"

if ! [[ -e "$innerdir/recon-00000.png" ]]
then
	innerdir="$(ls -d $1/????? | tail -1 )"
fi

row="`mktemp -d`"
for i in $(ls $innerdir/image-*.png | head -100)
do
	echo
	echo
	echo
	echo "HELLO"
	echo "$i"
	dir="$(dirname "$i")"
	suffix="${i##*-}"

	image="$i"
	diff="$dir/diff-$suffix"
	recon="$dir/recon-$suffix"

	width=$(pngtopnm "$image" | pnmtoTpnm | awk 'NR==1{print $2/2}')


	pnmcat -tb <(
		pnmarith -add <( pngtopnm "$image" | pnmcut -left 0 -width $width | pnmtoTpnm | awk 'NR==1;NR>1{print $1,  0,  0}') \
		              <( pngtopnm "$image" | pnmcut -left $width          | pnmtoTpnm | awk 'NR==1;NR>1{print  0, $2, $3}') \
	)  <(
		pnmarith -add <( pngtopnm "$recon" | pnmcut -left 0 -width $width | pnmtoTpnm | awk 'NR==1;NR>1{print $1,  0,  0}') \
		              <( pngtopnm "$recon" | pnmcut -left $width          | pnmtoTpnm | awk 'NR==1;NR>1{print  0, $2, $3}') \
	)  <(
		pnmarith -add <( pngtopnm "$diff" | pnmcut -left 0 -width $width | pnmtoTpnm | awk 'NR==1;NR>1{print $1,  0,  0}') \
		              <( pngtopnm "$diff" | pnmcut -left $width          | pnmtoTpnm | awk 'NR==1;NR>1{print  0, $2, $3}') \
	) | pnmpad -white -left=10 -right=10 -top=10 -bottom=30 > "$row/$suffix"
		   
done

num=$(ls $row | wc -l)
read _ w h _ < <(  cat $row/* | pnmtoTpnm | head -1 ) 

cols=`awk -vw=$w -vh=$h -vn=$num '
BEGIN{
	best=10000
	best_c = 0

	for(cols=1; cols <= 10; cols++){
		rows = int(n/cols + .9999999)


		width = cols * w
		height = rows * h

		err =  (log(width/height) - log(sqrt(2)))^2

		if(err < best) {
			best=err
			best_c = cols
		}
	}
	print best_c
}'`

{ echo -n 'pnmcat -tb ' ; ls $row/* | xargs -n $cols echo | xargs -d'\n' -Ix echo '<( pnmcat -lr ' x ')' | xargs echo ; }   | bash | pnmtopng | tee hax/montage.png | display -

rm -r "$row"
