#!/bin/bash

if [[ $# != 1 && $# != 2 ]]
then
	cat<<-FOOD
	Incorrect args 
	$0 <log_dir> [stride]
	FOOD
	exit 1
fi

rm -fr hax/log_paste
mkdir -p hax/log_paste


innerdir="$(ls -d $1/????? | tail -1 )"

for i in $(ls $innerdir/image-*.png)
do
	dir="$(dirname "$i")"
	suffix="${i##*-}"

	image="$i"
	diff="$dir/diff-$suffix"
	recon="$dir/recon-$suffix"

	pnmcat -tb <(pngtopnm "$image")  <(pngtopnm "$recon")  <(pngtopnm "$diff") | 
		   pnmpad -white -left=10 -right=10 -top=10 -bottom=30 > "hax/log_paste/$suffix"

	echo $i
		   
done

