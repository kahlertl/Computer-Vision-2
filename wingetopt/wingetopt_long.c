/*
 * Copyright (c) 1987, 1993, 1994, 1996
 *    The Regents of the University of California.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *    This product includes software developed by the University of
 *    California, Berkeley and its contributors.
 * 4. Neither the name of the University nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

#if defined FORCE_WINGETOPT || !defined __GNUC__

#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "getopt.h"

extern int      opterr;    /* if error message should be printed */
extern int      optind;    /* index into parent argv vector */
extern int      optopt;    /* character checked for validity */
extern int      optreset;    /* reset getopt */
extern char *optarg;    /* argument associated with option */

#define __P(x) x
#define _DIAGASSERT(x) assert(x)

static char * __progname __P((char *));
int getopt_internal __P((int, char * const *, const char *));

static char *
__progname(nargv0)
    char * nargv0;
{
    char * tmp;

    _DIAGASSERT(nargv0 != NULL);

    tmp = strrchr(nargv0, '/');
    if (tmp)
        tmp++;
    else
        tmp = nargv0;
    return(tmp);
}

#define    BADCH    (int)'?'
#define    BADARG    (int)':'
#define    EMSG    ""

/* 
 * The getopt() function parses the command-line arguments. Its arguments argc
 * and argv are the argument count and array as passed to the main() function on
 * program invocation. An element of argv that starts with '-' (and is not
 * exactly "-" or "--") is an option element. The characters of this element
 * (aside from the initial '-') are option characters. If getopt() is called
 * repeatedly, it returns successively each of the option characters from each of
 * the option elements.
 * 
 * The variable optind is the index of the next element to be processed in argv.
 * The system initializes this value to 1. The caller can reset it to 1 to
 * restart scanning of the same argv, or when scanning a new argument vector.
 * 
 * If getopt() finds another option character, it returns that character,
 * updating the external variable optind and a static variable nextchar so that
 * the next call to getopt() can resume the scan with the following option
 * character or argv-element.
 * 
 * If there are no more option characters, getopt() returns -1. Then optind is
 * the index in argv of the first argv-element that is not an option.
 * 
 * optstring is a string containing the legitimate option characters. If such a
 * character is followed by a colon, the option requires an argument, so getopt()
 * places a pointer to the following text in the same argv-element, or the text
 * of the following argv-element, in optarg. Two colons mean an option takes an
 * optional arg; if there is text in the current argv-element (i.e., in the same
 * word as the option name itself, for example, "-oarg"), then it is returned in
 * optarg, otherwise optarg is set to zero. This is a GNU extension. If optstring
 * contains W followed by a semicolon, then -W foo is treated as the long option
 * --foo. (The -W option is reserved by POSIX.2 for implementation extensions.)
 * This behavior is a GNU extension, not available with libraries before glibc 2.
 * 
 * By default, getopt() permutes the contents of argv as it scans, so that
 * eventually all the nonoptions are at the end. Two other modes are also
 * implemented. If the first character of optstring is '+' or the environment
 * variable POSIXLY_CORRECT is set, then option processing stops as soon as a
 * nonoption argument is encountered. If the first character of optstring is '-',
 * then each nonoption argv-element is handled as if it were the argument of an
 * option with character code 1. (This is used by programs that were written to
 * expect options and other argv-elements in any order and that care about the
 * ordering of the two.) The special argument "--" forces an end of option-
 * scanning regardless of the scanning mode.
 * 
 * If getopt() does not recognize an option character, it prints an error message
 * to stderr, stores the character in optopt, and returns '?'. The calling
 * program may prevent the error message by setting opterr to 0.
 * 
 * If getopt() finds an option character in argv that was not included in
 * optstring, or if it detects a missing option argument, it returns '?' and sets
 * the external variable optopt to the actual option character. If the first
 * character (following any optional '+' or '-' described above) of optstring is
 * a colon (':'), then getopt() returns ':' instead of '?' to indicate a missing
 * option argument. If an error was detected, and the first character of
 * optstring is not a colon, and the external variable opterr is nonzero (which
 * is the default), getopt() prints an error message.
 */
int
getopt_internal(nargc, nargv, ostr)
    int nargc;
    char * const *nargv;
    const char *ostr;
{
    static char *place = EMSG;        /* option letter processing */
    char *oli;                /* option letter list index */

    _DIAGASSERT(nargv != NULL);
    _DIAGASSERT(ostr != NULL);

    if (optreset || !*place) {        /* update scanning pointer */
        optreset = 0;
        if (optind >= nargc || *(place = nargv[optind]) != '-') {
            place = EMSG;
            return (-1);
        }
        if (place[1] && *++place == '-') {    /* found "--" */
            /* ++optind; */
            place = EMSG;
            return (-2);
        }
    }                    /* option letter okay? */
    if ((optopt = (int)*place++) == (int)':' ||
        !(oli = strchr(ostr, optopt))) {
        /*
         * if the user didn't specify '-' as an option,
         * assume it means -1.
         */
        if (optopt == (int)'-')
            return (-1);
        if (!*place)
            ++optind;
        if (opterr && *ostr != ':')
            (void)fprintf(stderr,
                "%s: illegal option -- %c\n", __progname(nargv[0]), optopt);
        return (BADCH);
    }
    if (*++oli != ':') {            /* don't need argument */
        optarg = NULL;
        if (!*place)
            ++optind;
    } else {                /* need an argument */
        if (*place)            /* no white space */
            optarg = place;
        else if (nargc <= ++optind) {    /* no arg */
            place = EMSG;
            if ((opterr) && (*ostr != ':'))
                (void)fprintf(stderr,
                    "%s: option requires an argument -- %c\n",
                    __progname(nargv[0]), optopt);
            return (BADARG);
        } else                /* white space */
            optarg = nargv[optind];
        place = EMSG;
        ++optind;
    }
    return (optopt);            /* dump back option letter */
}

#if 0
/*
 * getopt --
 *    Parse argc/argv argument vector.
 */
int
getopt2(nargc, nargv, ostr)
    int nargc;
    char * const *nargv;
    const char *ostr;
{
    int retval;

    if ((retval = getopt_internal(nargc, nargv, ostr)) == -2) {
        retval = -1;
        ++optind; 
    }
    return(retval);
}
#endif


/*
 * The getopt_long() function works like getopt() except that it also accepts
 * long options, started with two dashes. (If the program accepts only long
 * options, then optstring should be specified as an empty string (""), not
 * NULL.) Long option names may be abbreviated if the abbreviation is unique or
 * is an exact match for some defined option. A long option may take a parameter,
 * of the form --arg=param or --arg param.
 */
int getopt_long(int nargc, char * const nargv[],
                const char *options,
                const struct option *long_options, int *index)
{
    int retval;

    _DIAGASSERT(nargv != NULL);
    _DIAGASSERT(options != NULL);
    _DIAGASSERT(long_options != NULL);
    /* index may be NULL */

    if ((retval = getopt_internal(nargc, nargv, options)) == -2) {
        char *current_argv = nargv[optind++] + 2, *has_equal;
        int i, current_argv_len, match = -1;

        if (*current_argv == '\0') {
            return(-1);
        }
        if ((has_equal = strchr(current_argv, '=')) != NULL) {
            current_argv_len = has_equal - current_argv;
            has_equal++;
        } else
            current_argv_len = strlen(current_argv);

        for (i = 0; long_options[i].name; i++) { 
            if (strncmp(current_argv, long_options[i].name, current_argv_len))
                continue;

            if (strlen(long_options[i].name) == (unsigned)current_argv_len) { 
                match = i;
                break;
            }
            if (match == -1)
                match = i;
        }
        if (match != -1) {
            if (long_options[match].has_arg == required_argument ||
                long_options[match].has_arg == optional_argument) {
                if (has_equal)
                    optarg = has_equal;
                else
                    optarg = nargv[optind++];
            }
            if ((long_options[match].has_arg == required_argument)
                && (optarg == NULL)) {
                /*
                 * Missing argument, leading :
                 * indicates no error should be generated
                 */
                if ((opterr) && (*options != ':'))
                    (void)fprintf(stderr,
                      "%s: option requires an argument -- %s\n",
                      __progname(nargv[0]), current_argv);
                return (BADARG);
            }
        } else { /* No matching argument */
            if ((opterr) && (*options != ':'))
                (void)fprintf(stderr,
                    "%s: illegal option -- %s\n", __progname(nargv[0]), current_argv);
            return (BADCH);
        }
        if (long_options[match].flag) {
            *long_options[match].flag = long_options[match].val;
            retval = 0;
        } else 
            retval = long_options[match].val;
        if (index)
            *index = match;
    }
    return(retval);
}

#endif
