_OBJ=util.o main.o
CC=g++
CFLAGS=-Wall -c -std=c++11 -I.
SRCDIR=ps5
ODIR=tmp
OBJ=$(patsubst %,$(ODIR)/%,$(_OBJ))
_INC=
IDIR=include
INC=$(patsubst %,$(IDIR)/%,$(_INC))
LIBS=-lm

.PHONY: clean

cr: main clean

main: $(OBJ)
	$(CC) -Wall -O2 -o main $^ $(LIBS)

$(ODIR)/%.o: $(SRCDIR)/%.cpp $(INC)
	$(CC)  $(CFLAGS) $< -o $@

clean: 
	rm -f *.o $(ODIR)/*
