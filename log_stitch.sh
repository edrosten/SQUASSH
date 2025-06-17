#!/bin/bash

if [[ $# != 1 && $# != 2 ]]
then
	cat<<-FOOD
	Incorrect args 
	$0 <log_dir> [stride]
	FOOD
	exit 1
fi

cols=10

innerdir="$1"
skip="${2:-20}"



rows="`mktemp -d`"

for innerdir in $(ls -d $1/????? | awk -vskip=$skip 'NR%skip==0' )
do

	row="`mktemp -d`"
	for i in $(ls $innerdir/image-*.png | head -20)
	do
		dir="$(dirname "$i")"
		suffix="${i##*-}"

		image="$i"
		diff="$dir/diff-$suffix"
		recon="$dir/recon-$suffix"

		pnmcat -tb <(pngtopnm "$image")  <(pngtopnm "$recon")  <(pngtopnm "$diff") | 
			   pnmpad -white -left=10 -right=10 -top=10 -bottom=30 > "$row/$suffix"
			   
	done
	pnmcat -lr $row/* > "$rows/${innerdir##*/}"
	rm -r "$row"
done

pnmcat -tb $rows/* | pnmtopng | tee hax/stitch.png | display -
rm -r "$rows"
