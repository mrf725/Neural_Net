#!/bin/sh
#
# run neural net bin

echo Starting Serial
echo
echo Target MSE = 0.1
export TARGET_MSE=0.1
./neural_net_serial_contiguous.exe 400 25 104 26 2 400 .5 Testing.txt Training.txt Targets.txt
./neural_net_serial_contiguous.exe 400 25 104 26 2 400 .5 Testing.txt Training.txt Targets.txt
./neural_net_serial_contiguous.exe 400 25 104 26 2 400 .5 Testing.txt Training.txt Targets.txt
echo
echo Target MSE = 0.05
export TARGET_MSE=0.05
./neural_net_serial_contiguous.exe 400 25 104 26 2 400 .5 Testing.txt Training.txt Targets.txt
./neural_net_serial_contiguous.exe 400 25 104 26 2 400 .5 Testing.txt Training.txt Targets.txt
./neural_net_serial_contiguous.exe 400 25 104 26 2 400 .5 Testing.txt Training.txt Targets.txt
echo
echo Target MSE = 0.01
export TARGET_MSE=0.01
./neural_net_serial_contiguous.exe 400 25 104 26 2 400 .5 Testing.txt Training.txt Targets.txt
./neural_net_serial_contiguous.exe 400 25 104 26 2 400 .5 Testing.txt Training.txt Targets.txt
./neural_net_serial_contiguous.exe 400 25 104 26 2 400 .5 Testing.txt Training.txt Targets.txt


echo
echo
echo

echo Starting Parallel: Data
echo NP = 2
echo Target MSE = 0.1
export TARGET_MSE=0.1
mpirun -np 2 neural_net.exe 400 25 104 26 2 400 .5 Testing.txt Training.txt Targets.txt
mpirun -np 2 neural_net.exe 400 25 104 26 2 400 .5 Testing.txt Training.txt Targets.txt
mpirun -np 2 neural_net.exe 400 25 104 26 2 400 .5 Testing.txt Training.txt Targets.txt
echo
echo Target MSE = 0.05
export TARGET_MSE=0.05
mpirun -np 2 neural_net.exe 400 25 104 26 2 400 .5 Testing.txt Training.txt Targets.txt
mpirun -np 2 neural_net.exe 400 25 104 26 2 400 .5 Testing.txt Training.txt Targets.txt
mpirun -np 2 neural_net.exe 400 25 104 26 2 400 .5 Testing.txt Training.txt Targets.txt
echo
echo Target MSE = 0.01
export TARGET_MSE=0.01
mpirun -np 2 neural_net.exe 400 25 104 26 2 400 .5 Testing.txt Training.txt Targets.txt
mpirun -np 2 neural_net.exe 400 25 104 26 2 400 .5 Testing.txt Training.txt Targets.txt
mpirun -np 2 neural_net.exe 400 25 104 26 2 400 .5 Testing.txt Training.txt Targets.txt

echo
echo

echo NP = 4
echo Target MSE = 0.1
export TARGET_MSE=0.1
mpirun -np 4 neural_net.exe 400 25 104 26 2 400 .5 Testing.txt Training.txt Targets.txt
mpirun -np 4 neural_net.exe 400 25 104 26 2 400 .5 Testing.txt Training.txt Targets.txt
mpirun -np 4 neural_net.exe 400 25 104 26 2 400 .5 Testing.txt Training.txt Targets.txt
echo
echo Target MSE = 0.05
export TARGET_MSE=0.05
mpirun -np 4 neural_net.exe 400 25 104 26 2 400 .5 Testing.txt Training.txt Targets.txt
mpirun -np 4 neural_net.exe 400 25 104 26 2 400 .5 Testing.txt Training.txt Targets.txt
mpirun -np 4 neural_net.exe 400 25 104 26 2 400 .5 Testing.txt Training.txt Targets.txt
echo
echo Target MSE = 0.01
export TARGET_MSE=0.01
mpirun -np 4 neural_net.exe 400 25 104 26 2 400 .5 Testing.txt Training.txt Targets.txt
mpirun -np 4 neural_net.exe 400 25 104 26 2 400 .5 Testing.txt Training.txt Targets.txt
mpirun -np 4 neural_net.exe 400 25 104 26 2 400 .5 Testing.txt Training.txt Targets.txt

echo
echo

echo NP = 8
echo Target MSE = 0.1
export TARGET_MSE=0.1
mpirun -np 8 neural_net.exe 400 25 104 26 2 400 .5 Testing.txt Training.txt Targets.txt
mpirun -np 8 neural_net.exe 400 25 104 26 2 400 .5 Testing.txt Training.txt Targets.txt
mpirun -np 8 neural_net.exe 400 25 104 26 2 400 .5 Testing.txt Training.txt Targets.txt
echo
echo Target MSE = 0.05
export TARGET_MSE=0.05
mpirun -np 8 neural_net.exe 400 25 104 26 2 400 .5 Testing.txt Training.txt Targets.txt
mpirun -np 8 neural_net.exe 400 25 104 26 2 400 .5 Testing.txt Training.txt Targets.txt
mpirun -np 8 neural_net.exe 400 25 104 26 2 400 .5 Testing.txt Training.txt Targets.txt
echo
echo Target MSE = 0.01
export TARGET_MSE=0.01
mpirun -np 8 neural_net.exe 400 25 104 26 2 400 .5 Testing.txt Training.txt Targets.txt
mpirun -np 8 neural_net.exe 400 25 104 26 2 400 .5 Testing.txt Training.txt Targets.txt
mpirun -np 8 neural_net.exe 400 25 104 26 2 400 .5 Testing.txt Training.txt Targets.txt

echo
echo
echo

echo Starting Parallel: Node
echo Threads = 2
echo Target MSE = 0.1
export TARGET_MSE=0.1
export OMP_NUM_THREADS=2
mpirun -np 1 neural_net.exe 400 25 104 26 2 400 .5 Testing.txt Training.txt Targets.txt
mpirun -np 1 neural_net.exe 400 25 104 26 2 400 .5 Testing.txt Training.txt Targets.txt
mpirun -np 1 neural_net.exe 400 25 104 26 2 400 .5 Testing.txt Training.txt Targets.txt
echo
echo Target MSE = 0.05
export TARGET_MSE=0.05
export OMP_NUM_THREADS=2
mpirun -np 1 neural_net.exe 400 25 104 26 2 400 .5 Testing.txt Training.txt Targets.txt
mpirun -np 1 neural_net.exe 400 25 104 26 2 400 .5 Testing.txt Training.txt Targets.txt
mpirun -np 1 neural_net.exe 400 25 104 26 2 400 .5 Testing.txt Training.txt Targets.txt
echo
echo Target MSE = 0.01
export TARGET_MSE=0.01
export OMP_NUM_THREADS=2
mpirun -np 1 neural_net.exe 400 25 104 26 2 400 .5 Testing.txt Training.txt Targets.txt
mpirun -np 1 neural_net.exe 400 25 104 26 2 400 .5 Testing.txt Training.txt Targets.txt
mpirun -np 1 neural_net.exe 400 25 104 26 2 400 .5 Testing.txt Training.txt Targets.txt

echo
echo

echo Threads = 4
echo Target MSE = 0.1
export TARGET_MSE=0.1
export OMP_NUM_THREADS=4
mpirun -np 1 neural_net.exe 400 25 104 26 2 400 .5 Testing.txt Training.txt Targets.txt
mpirun -np 1 neural_net.exe 400 25 104 26 2 400 .5 Testing.txt Training.txt Targets.txt
mpirun -np 1 neural_net.exe 400 25 104 26 2 400 .5 Testing.txt Training.txt Targets.txt
echo
echo Target MSE = 0.05
export TARGET_MSE=0.05
export OMP_NUM_THREADS=4
mpirun -np 1 neural_net.exe 400 25 104 26 2 400 .5 Testing.txt Training.txt Targets.txt
mpirun -np 1 neural_net.exe 400 25 104 26 2 400 .5 Testing.txt Training.txt Targets.txt
mpirun -np 1 neural_net.exe 400 25 104 26 2 400 .5 Testing.txt Training.txt Targets.txt
echo
echo Target MSE = 0.01
export TARGET_MSE=0.01
export OMP_NUM_THREADS=4
mpirun -np 1 neural_net.exe 400 25 104 26 2 400 .5 Testing.txt Training.txt Targets.txt
mpirun -np 1 neural_net.exe 400 25 104 26 2 400 .5 Testing.txt Training.txt Targets.txt
mpirun -np 1 neural_net.exe 400 25 104 26 2 400 .5 Testing.txt Training.txt Targets.txt

echo
echo

echo Threads = 8
echo Target MSE = 0.1
export TARGET_MSE=0.1
export OMP_NUM_THREADS=8
mpirun -np 1 neural_net.exe 400 25 104 26 2 400 .5 Testing.txt Training.txt Targets.txt
mpirun -np 1 neural_net.exe 400 25 104 26 2 400 .5 Testing.txt Training.txt Targets.txt
mpirun -np 1 neural_net.exe 400 25 104 26 2 400 .5 Testing.txt Training.txt Targets.txt
echo
echo Target MSE = 0.05
export TARGET_MSE=0.05
export OMP_NUM_THREADS=8
mpirun -np 1 neural_net.exe 400 25 104 26 2 400 .5 Testing.txt Training.txt Targets.txt
mpirun -np 1 neural_net.exe 400 25 104 26 2 400 .5 Testing.txt Training.txt Targets.txt
mpirun -np 1 neural_net.exe 400 25 104 26 2 400 .5 Testing.txt Training.txt Targets.txt
echo
echo Target MSE = 0.01
export TARGET_MSE=0.01
export OMP_NUM_THREADS=8
mpirun -np 1 neural_net.exe 400 25 104 26 2 400 .5 Testing.txt Training.txt Targets.txt
mpirun -np 1 neural_net.exe 400 25 104 26 2 400 .5 Testing.txt Training.txt Targets.txt
mpirun -np 1 neural_net.exe 400 25 104 26 2 400 .5 Testing.txt Training.txt Targets.txt


echo
echo
echo

echo Sync Rate
echo Sync = 100
echo NP = 2
echo Target MSE = 0.1
export OMP_NUM_THREADS=1
export TARGET_MSE=0.1
export SYNC_RATE=100
mpirun -np 2 neural_net.exe 400 25 104 26 2 400 .5 Testing.txt Training.txt Targets.txt
mpirun -np 2 neural_net.exe 400 25 104 26 2 400 .5 Testing.txt Training.txt Targets.txt
mpirun -np 2 neural_net.exe 400 25 104 26 2 400 .5 Testing.txt Training.txt Targets.txt
echo
echo Sync = 500
echo NP = 2
echo Target MSE = 0.1
export TARGET_MSE=0.1
export SYNC_RATE=500
mpirun -np 2 neural_net.exe 400 25 104 26 2 400 .5 Testing.txt Training.txt Targets.txt
mpirun -np 2 neural_net.exe 400 25 104 26 2 400 .5 Testing.txt Training.txt Targets.txt
mpirun -np 2 neural_net.exe 400 25 104 26 2 400 .5 Testing.txt Training.txt Targets.txt
echo
echo Sync = 1000
echo NP = 2
echo Target MSE = 0.1
export TARGET_MSE=0.1
export SYNC_RATE=1000
mpirun -np 2 neural_net.exe 400 25 104 26 2 400 .5 Testing.txt Training.txt Targets.txt
mpirun -np 2 neural_net.exe 400 25 104 26 2 400 .5 Testing.txt Training.txt Targets.txt
mpirun -np 2 neural_net.exe 400 25 104 26 2 400 .5 Testing.txt Training.txt Targets.txt


echo
echo
echo

echo Parallel: Data and Node

echo Target MSE = 0.05
echo NP = 2
echo Threads = 2
export TARGET_MSE=0.05
export OMP_NUM_THREADS=2
export SYNC_RATE=100
mpirun -np 2 neural_net.exe 400 25 104 26 2 400 .5 Testing.txt Training.txt Targets.txt
echo
echo NP = 4
echo Threads = 4
export TARGET_MSE=0.05
export OMP_NUM_THREADS=4
mpirun -np 4 neural_net.exe 400 25 104 26 2 400 .5 Testing.txt Training.txt Targets.txt
echo
echo NP = 8
echo Threads = 8
export TARGET_MSE=0.05
export OMP_NUM_THREADS=8
mpirun -np 8 neural_net.exe 400 25 104 26 2 400 .5 Testing.txt Training.txt Targets.txt
echo
echo NP = 2
echo Threads = 4
export TARGET_MSE=0.05
export OMP_NUM_THREADS=4
mpirun -np 2 neural_net.exe 400 25 104 26 2 400 .5 Testing.txt Training.txt Targets.txt
