/**
 * POSIX getopt for Windows
 * 
 * AT&T Public License
 * 
 * Code given out at the 1985 UNIFORUM conference in Dallas.
 * 
 * source: http://note.sonots.com/Comp/CompLang/cpp/getopt.html
 */

#if defined __GNUC__ && !defined FORCE_WINGETOPT
#include <getopt.h>
#else

#ifndef _WINGETOPT_H
#define _WINGETOPT_H
#define _GETOPT_H // define to prevent <getopt.h> to be included

#ifdef __cplusplus
extern "C" {
#endif

extern int opterr;        /* if error message should be printed */
extern int optind;        /* index into parent argv vector */
extern int optopt;        /* character checked for validity */
extern int optreset;        /* reset getopt */
extern char *optarg;        /* argument associated with option */

struct option
{
  const char *name;
  int has_arg;
  int *flag;
  int val;
};

#define no_argument       0
#define required_argument 1
#define optional_argument 2

int getopt(int, char**, char*);
int getopt_long(int argc, char * const argv[],
                const char *optstring,
                const struct option *longopts, int *longindex);

#ifdef __cplusplus
}
#endif

#endif  /* _WINGETOPT_H */
#endif  /* __GNUC__ && FORCE_WINGETOPT */
