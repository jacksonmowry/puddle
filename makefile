# TODO this is a garbage makefile lol

all: bin/generate_reservoir bin/data_preprocessing bin/classify bin/grade

bin/generate_reservoir: scripts/generate_reservoir.c
	$(CC) scripts/generate_reservoir.c -o bin/generate_reservoir -lm

bin/data_preprocessing: scripts/data_preprocessing.c
	$(CC) scripts/data_preprocessing.c -o bin/data_preprocessing -lm

bin/classify: src/reservoir_classify.cpp framework-open/lib/libframework.a framework-open/obj/risp.o framework-open/obj/risp_static.o
	$(CXX) $(CXXFLAGS) src/reservoir_classify.cpp framework-open/lib/libframework.a framework-open/obj/risp* -o bin/classify -Iframework-open/include -O2

bin/grade: src/reservoir_grade.cpp framework-open/lib/libframework.a framework-open/obj/risp.o framework-open/obj/risp_static.o
	$(CXX) $(CXXFLAGS) src/reservoir_grade.cpp framework-open/lib/libframework.a framework-open/obj/risp* -o bin/grade -Iframework-open/include -O2

framework-open/lib/libframework.a:
	(cd framework-open; make)

framework-open/obj/risp.o:
	(cd framework-open)

framework-open/obj/risp_static.o:
	(cd framework-open)

clean:
	rm bin/*; (cd framework-open; make clean)
