#!/bin/bash
f="$1"
shift

awk "$@" -f loss.awk log/"$f"/loss.txt
