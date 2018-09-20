#! /bin/bash
git clone https://github.com/uWebSockets/uWebSockets
cd uWebSockets
git checkout e94b6e1
mkdir build
cd build
cmake ..
make
make install
cd ../..
rm -r uWebSockets
