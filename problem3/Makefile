# INCLUDE_DIRS = -I/opt/intel/compilers_and_libraries_2020.0.166/linux/mpi/intel64/include/
# LIB_DIRS = -L/opt/intel/compilers_and_libraries_2020.0.166/linux/mpi/intel64/lib/debug -L/opt/intel/compilers_and_libraries_2020.0.166/linux/mpi/intel64/lib
#LIB_DIRS = -L/opt/intel/compilers_and_libraries_2020.0.166/linux/mpi/intel64/lib/release -L/opt/intel/compilers_and_libraries_2020.0.166/linux/mpi/intel64/lib
INCLUDE_DIRS = 
CC = mpicc
CXX = mpicxx
GCC = gcc

CDEFS=
CFLAGS= -g -Wall $(INCLUDE_DIRS) $(CDEFS)
LIBS=

PRODUCT= problem1 problem2-a problem2-b problem3
HFILES= 
CFILES= problem1.c problem2-a.c problem2-b.c problem3.c timeinterp.c

SRCS= ${HFILES} ${CFILES}
OBJS= ${CFILES:.c=.o}

all:	${PRODUCT}

clean:
	-rm -f *.o *.NEW *~
	-rm -f ${PRODUCT} ${DERIVED} ${GARBAGE}

problem1:	problem1.c
	$(CC) $(CFLAGS) -o $@ problem1.c $(LIB_DIRS) -lm

problem2-a:	problem2-a.c
	$(CC) $(CFLAGS) -o $@ problem2-a.c $(LIB_DIRS) -lm

problem2-b:	problem2-b.c
	$(CC) $(CFLAGS) -o $@ problem2-b.c $(LIB_DIRS) -lm

problem3:	problem3.c
	$(CC) $(CFLAGS) -o $@ problem3.c timeinterp.c $(LIB_DIRS) -lm


