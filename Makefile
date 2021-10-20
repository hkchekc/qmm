CC=g++-8
CFLAGS=-Wall -O3 -fopenmp -mavx -ffast-math -ftree-vectorize -Wextra -c -std=c++17 -I. -I/usr/local/include
SRCDIR=hugget
_OBJ=$(SRCDIR).o main.o 
ODIR=tmp
OBJ=$(patsubst %,$(ODIR)/%,$(_OBJ))
_INC= main $(SRCDIR)
IDIR=include
INC=$(patsubst %,$(SRCDIR)/$(IDIR)/%,$(_INC))
LIBS=-lm -L/usr/local/lib -lgsl
TARDIR=build

.PHONY: clean

cr: main clean

main: $(OBJ)
	$(CC) -Wall -O3 -fopenmp $^ $(LIBS) -o $(TARDIR)/$(SRCDIR)_$@ -lomp

$(ODIR)/%.o: $(SRCDIR)/%.cpp $(INC)
	$(CC)  $(CFLAGS) $< -o $@

clean: 
	rm -f *.o $(ODIR)/*
	$(TARDIR)/$(SRCDIR)_main
#	rm -f $(TARDIR)/*
