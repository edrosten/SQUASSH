set -e
#mkdir hax/meshes3

j=0
ls log/1721115624-7dd345ea35074b5fba53489b4c12ac512b3b5a61/run-000-phase_0/final_model.txt |
	awk 'NR%20==0' |
while read i
do

	printf 'python render_marching_cubes.py %s > hax/meshes/%04i.ply ; echo %i \n' $i $j $j
	: $((j++))
done

