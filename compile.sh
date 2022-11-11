#!/bin/sh

./format.sh

cd bin/

rm -rf *

cmake ..

make
