# TODO this is a garbage makefile lol
CFLAGS=-std=c2x
unexport CFLAGS

all: bin/generate_reservoir bin/data_preprocessing bin/classify bin/grade bin/control bin/control_crisp

bin/generate_reservoir: scripts/generate_reservoir.c
	$(CC) $(CFLAGS) scripts/generate_reservoir.c -o bin/generate_reservoir -lm

bin/data_preprocessing: scripts/data_preprocessing.c
	$(CC) $(CFLAGS) scripts/data_preprocessing.c -o bin/data_preprocessing -lm

bin/classify: src/reservoir_classify.cpp framework-open/lib/libframework.a framework-open/obj/risp.o framework-open/obj/risp_static.o
	$(CXX) $(CXXFLAGS) src/reservoir_classify.cpp framework-open/lib/libframework.a framework-open/obj/risp* -o bin/classify -Iframework-open/include -O2

bin/grade: src/reservoir_grade.cpp framework-open/lib/libframework.a framework-open/obj/risp.o framework-open/obj/risp_static.o
	$(CXX) $(CXXFLAGS) src/reservoir_grade.cpp framework-open/lib/libframework.a framework-open/obj/risp* -o bin/grade -Iframework-open/include -O2

bin/control: src/reservoir_control.cpp framework-open/lib/libframework.a framework-open/obj/risp.o framework-open/obj/risp_static.o
	$(CXX) $(CXXFLAGS) src/reservoir_control.cpp framework-open/lib/libframework.a framework-open/obj/risp* -o bin/control -Iframework-open/include -O2

bin/control_crisp: src/reservoir_control.cpp framework-open/lib/libframework.a framework-open/obj/crisp.o framework-open/obj/crisp_static.o
	$(CXX) $(CXXFLAGS) src/reservoir_control.cpp framework-open/lib/libframework.a framework-open/obj/crisp* -o bin/control_crisp -Iframework-open/include -O2

framework-open/lib/libframework.a:
	(cd framework-open; make)

framework-open/obj/risp.o:
	(cd framework-open)

framework-open/obj/risp_static.o:
	(cd framework-open)

framework-open/obj/crisp.o:
	(cd framework-open)

framework-open/obj/crisp_static.o:
	(cd framework-open)

clean:
	rm bin/*; (cd framework-open; make clean)
