CC		:= g++ # This is the main compiler
# CC := clang --analyze # and comment out the linker last line for sanity
SRCDIR	:= src
LIBDIR	:= lib
BUILDDIR:= build
TARGET	:= bin/main

SRCEXT	:= cpp
SOURCES	:= $(shell find $(SRCDIR) -type f -name *.$(SRCEXT))
OBJECTS	:= $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(SOURCES:.$(SRCEXT)=.o))
CFLAGS	:= -g # -Wall
#LIB		:= $(shell find $(LIBDIR) -type f -name *.$(SRCEXT))
#INC		:= -I include -L include/dlib/all/source.cpp
INC		:= -Iinclude/

$(TARGET): $(OBJECTS)
	@echo " Linking..."
	@echo " $(CC) $^ -o $(TARGET) $(LIB)"; $(CC) $^ -o $(TARGET) $(LIB) -Iinclude/dlib -Linclude/dlib/all/source.cpp

$(BUILDDIR)/%.o: $(SRCDIR)/%.$(SRCEXT)
	@mkdir -p $(BUILDDIR)
	@echo " $(CC) $(CFLAGS) $(INC) -c -o $@ $<"; $(CC) $(CFLAGS) $(INC) -c -o $@ $<

clean:
	@echo " Cleaning..."; 
	@echo " $(RM) -r $(BUILDDIR) $(TARGET)"; $(RM) -r $(BUILDDIR) $(TARGET)

## Tests
#tester:
#  $(CC) $(CFLAGS) test/tester.cpp $(INC) $(LIB) -o bin/tester

## Spikes
#ticket:
#  $(CC) $(CFLAGS) spikes/ticket.cpp $(INC) $(LIB) -o bin/ticket

.PHONY: clean
