#!/bin/sh

res=$(find . -type f \( ! -path "./.git/*" ! -path "./bin/*" \) -prune \
         -regextype posix-extended -regex ".*\.origin\.(h|cxx|cpp)$")

for f in $res
do
	echo "Formatting === $f ==="
	r=$(echo $f | sed "s/\.origin//")
	clang-format $f >$r
done;