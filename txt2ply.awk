!/^#/ && $4 > .5{a[i++]=$1 " " $2 " " $3}

END{
	print "ply"
	print "format ascii 1.0"
	print "element vertex " i
	print "property float x"
	print "property float y"
	print "property float z"
	print "end_header"
	for(j=0; j < i; j++)
		print(a[j])
}
