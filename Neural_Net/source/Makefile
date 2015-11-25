all: clean neural_net neural_net_serial neural_net_serial_contiguous

neural_net: neural_net.c
	mpicc -g -o neural_net neural_net.c -fopenmp

neural_net_serial: neural_net_serial.c
	gcc -g -o neural_net_serial neural_net_serial.c
	
neural_net_serial_contiguous: neural_net_serial_contiguous.c
	gcc -g -o neural_net_serial_contiguous neural_net_serial_contiguous.c

clean:
	rm -f neural_net
	rm -f neural_net_serial
	rm -f neural_net_serial_contiguous
