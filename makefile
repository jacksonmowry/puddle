CFLAGS=-std=c2x
unexport CFLAGS

all: bin/generate_reservoir \
	bin/data_preprocessing \
	bin/classify \
	bin/grade \
	bin/control \
	bin/control_crisp

bin/generate_reservoir: scripts/generate_reservoir.c
	$(CC) $(CFLAGS) $^ -o $@ -lm

bin/data_preprocessing: scripts/data_preprocessing.c
	$(CC) $(CFLAGS) $^ -o $@ -lm

bin/classify: src/reservoir_classify.cpp framework-open/lib/libframework.a framework-open/obj/risp.o framework-open/obj/risp_static.o obj/encoder.o
	$(CXX) $(CXXFLAGS) $^ -o $@ -Iinclude -Iframework-open/include

bin/grade: src/reservoir_grade.cpp framework-open/lib/libframework.a framework-open/obj/risp.o framework-open/obj/risp_static.o obj/encoder.o
	$(CXX) $(CXXFLAGS) $^ -o $@ -Iinclude -Iframework-open/include

bin/control: src/reservoir_control.cpp framework-open/lib/libframework.a framework-open/obj/risp.o framework-open/obj/risp_static.o obj/encoder.o
	$(CXX) $(CXXFLAGS) $^ -o $@ -Iinclude -Iframework-open/include

bin/control_crisp: src/reservoir_control.cpp framework-open/lib/libframework.a framework-open/obj/crisp.o framework-open/obj/crisp_static.o obj/encoder.o
	$(CXX) $(CXXFLAGS) $^ -o $@ -Iinclude -Iframework-open/include

obj/encoder.o: src/encoder.cpp
	$(CXX) $(CXXFLAGS) $^ -Iinclude -Iframework-open/include -c -o $@

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
