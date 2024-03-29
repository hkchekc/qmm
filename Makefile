CC=g++-8
CFLAGS=-Wall -O3 -fopenmp -mavx -ffast-math -ftree-vectorize -Wextra -c -std=c++17 -I. -I/usr/local/include
DEBUG_FLAGS = -g
SRCDIR=bkm
SUPPORT=aiyagari
ODIR=tmp
OBJ=$(patsubst %,$(ODIR)/%,$(_OBJ))
ifeq ($(SUPPORT),)
	_OBJ=$(SRCDIR).o main.o util.o
	_INC=$(SRCDIR).hpp util.hpp
else
	_OBJ=$(SRCDIR).o main.o util.o $(foreach s, $(SUPPORT), $(s)).o
	_INC=$(SRCDIR).hpp util.hpp $(foreach s, $(SUPPORT), $(s)).hpp
endif
IDIR=include
INC=$(patsubst %, $(IDIR)/%,$(_INC))
LIBS=-lm -L/usr/local/lib -lgsl
TARDIR=build

.PHONY: clean

cr: main clean

main: $(OBJ)
	$(CC) -Wall -O3 -fopenmp $^ $(LIBS) -o $(TARDIR)/$(SRCDIR)_$@ -lomp

$(ODIR)/%.o: $(SRCDIR)/%.cpp $(INC)
	$(CC)  $(CFLAGS) $(DEBUG_FLAGS) $< -o $@

$(ODIR)/%.o: $(foreach s, $(SUPPORT), $(s))/%.cpp $(INC)
	$(CC)  $(CFLAGS) $(DEBUG_FLAGS) $< -o $@

$(ODIR)/%.o: util/%.cpp $(INC)
	$(CC)  $(CFLAGS) $(DEBUG_FLAGS) $< -o $@

clean: 
	rm -f *.o $(ODIR)/*
	$(TARDIR)/$(SRCDIR)_main
#	rm -f $(TARDIR)/*
