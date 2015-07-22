option(HAVE_GETOPT_H     "Have the getopt library"  OFF)
option(HAVE_INTTYPES_H   "Have the file inttypes.h" ON)
option(HAVE_MEMORY_H     "Have the file memory.h"   ON)
option(HAVE_STDLIB_H     "Have the file stdlib.h"   ON)
option(HAVE_STRINGS_H    "Have the strings.h"       OFF)
option(HAVE_STRING_H     "Have the file string.h"   ON)
option(HAVE_STDC_HEADERS "Have std C header files"  ON)

set(LIB_SOURCES arg_dbl.c
                arg_end.c
                arg_file.c
                arg_int.c
                arg_lit.c
                arg_rem.c
                arg_str.c
                argtable2.c
                argtable2.h
                getopt.c
                getopt.h
                getopt1.c)

if(NOT WIN32)
    list(APPEND LIB_SOURCES arg_date.c arg_rex.c)
endif()


if(HAVE_GETOPT_H)
    set(COMPILE_FLAGS "${COMPILE_FLAGS} -DHAVE_GETOPT_H")
endif()

if(HAVE_INTTYPES_H)
    set(COMPILE_FLAGS "${COMPILE_FLAGS} -DHAVE_INTTYPES_H")
endif()

if(HAVE_MEMORY_H)
    set(COMPILE_FLAGS "${COMPILE_FLAGS} -DHAVE_MEMORY_H")
endif()

if(HAVE_STDLIB_H)
    set(COMPILE_FLAGS "${COMPILE_FLAGS} -DHAVE_STDLIB_H")
endif()

if(HAVE_STRINGS_H)
    set(COMPILE_FLAGS "${COMPILE_FLAGS} -DHAVE_STRINGS_H")
endif()

if(HAVE_STRING_H)
    set(COMPILE_FLAGS "${COMPILE_FLAGS} -DHAVE_STRING_H")
endif()

if(HAVE_STDC_HEADERS)
    set(COMPILE_FLAGS "${COMPILE_FLAGS} -DSTDC_HEADERS")
endif()

add_library(argtable2 STATIC ${LIB_SOURCES})
set_target_properties(argtable2 PROPERTIES
                                COMPILE_FLAGS "${COMPILE_FLAGS}")
