all: maoz

maoz:
	gcc -o  maoz maoz.c -lavutil -lavformat -lavcodec -lswscale -lz -lm `sdl-config --cflags --libs` > err  2>&1

clean:
	rm -rf maoz
