# TODO this is a garbage makefile lol

all: bin/generate_reservoir bin/classify

bin/generate_reservoir: scripts/generate_reservoir.c
	$(CC) scripts/generate_reservoir.c -o bin/generate_reservoir -lm

bin/classify: src/reservoir_classify.cpp framework-open/lib/libframework.a framework-open/obj/risp.o framework-open/obj/risp_static.o
	$(CXX) $(CXXFLAGS) src/reservoir_classify.cpp framework-open/lib/libframework.a framework-open/obj/risp* -o bin/classify -Iframework-open/include -O2

framework-open/lib/libframework.a:
	(cd framework-open; make -j4)

framework-open/obj/risp.o:
	(cd framework-open)

framework-open/obj/risp_static.o:
	(cd framework-open)

clean:
	rm bin/*; (cd framework-open; make clean)
