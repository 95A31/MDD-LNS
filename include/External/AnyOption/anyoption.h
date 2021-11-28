#pragma once

#define _CRT_SECURE_NO_WARNINGS /* Microsoft C/C++ Compiler: Disable C4996     \
                                   warnings for security-enhanced CRT          \
                                   functions */

#include <cstring>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <string>

#define COMMON_OPT 1
#define COMMAND_OPT 2
#define FILE_OPT 3
#define COMMON_FLAG 4
#define COMMAND_FLAG 5
#define FILE_FLAG 6

#define COMMAND_OPTION_TYPE 1
#define COMMAND_FLAG_TYPE 2
#define FILE_OPTION_TYPE 3
#define FILE_FLAG_TYPE 4
#define UNKNOWN_TYPE 5

#define DEFAULT_MAXOPTS 10
#define MAX_LONG_PREFIX_LENGTH 2

#define DEFAULT_MAXUSAGE 3
#define DEFAULT_MAXHELP 10

#define TRUE_FLAG "true"

using namespace std;

class AnyOption {

public: /* the public interface */
  AnyOption();

  explicit AnyOption(int maxoptions);
  explicit AnyOption(int maxoptions, int maxcharoptions);
  ~AnyOption();

  /*
   * following set methods specifies the
   * special characters and delimiters
   * if not set traditional defaults will be used
   */

  void setCommandPrefixChar(char _prefix);    /* '-' in "-w" */
  void setCommandLongPrefix(const char *_prefix); /* '--' in "--width" */
  void setFileCommentChar(char _comment);     /* '#' in shell scripts */
  void setFileDelimiterChar(char _delimiter); /* ':' in "width : 100" */

  /*
   * provide the input for the options
   * like argv[] for commnd line and the
   * option file name  to use;
   */

  void useCommandArgs(int _argc, char **_argv);
  void useFiileName(const char *_filename);

  /*
   * turn off the POSIX style options
   * this means anything starting with a '-' or "--"
   * will be considered a valid option
   * which also means you cannot add a bunch of
   * POIX options chars together like "-lr"  for "-l -r"
   *
   */

  void noPOSIX();

  /*
   * prints warning verbose if you set anything wrong
   */
  void setVerbose();

  /*
   * there are two types of options
   *
   * Option - has an associated value ( -w 100 )
   * Flag  - no value, just a boolean flag  ( -nogui )
   *
   * the options can be either a string ( GNU style )
   * or a character ( traditional POSIX style )
   * or both ( --width, -w )
   *
   * the options can be common to the command line and
   * the option file, or can belong only to either of
   * command line and option file
   *
   * following set methods, handle all the above
   * cases of options.
   */

  /* options command to command line and option file */
  void setOption(const char *opt_string);
  void setOption(char opt_char);
  void setOption(const char *opt_string, char opt_char);
  void setFlag(const char *opt_string);
  void setFlag(char opt_char);
  void setFlag(const char *opt_string, char opt_char);

  /* options read from command line only */
  void setCommandOption(const char *opt_string);
  void setCommandOption(char opt_char);
  void setCommandOption(const char *opt_string, char opt_char);
  void setCommandFlag(const char *opt_string);
  void setCommandFlag(char opt_char);
  void setCommandFlag(const char *opt_string, char opt_char);

  /* options read from an option file only  */
  void setFileOption(const char *opt_string);
  void setFileOption(char opt_char);
  void setFileOption(const char *opt_string, char opt_char);
  void setFileFlag(const char *opt_string);
  void setFileFlag(char opt_char);
  void setFileFlag(const char *opt_string, char opt_char);

  /*
   * process the options, registered using
   * useCommandArgs() and useFileName();
   */
  void processOptions();
  void processCommandArgs();
  void processCommandArgs(int max_args);
  bool processFile();

  /*
   * process the specified options
   */
  void processCommandArgs(int _argc, char **_argv);
  void processCommandArgs(int _argc, char **_argv, int max_args);
  bool processFile(const char *_filename);

  /*
   * get the value of the options
   * will return NULL if no value is set
   */
  char *getValue(const char *_option);
  bool getFlag(const char *_option);
  char *getValue(char _optchar);
  bool getFlag(char _optchar);

  /*
   * Print Usage
   */
  void printUsage();
  void printAutoUsage();
  void addUsage(const char *line);
  void printHelp();
  /* print auto usage printing for unknown options or flag */
  void autoUsagePrint(bool flag);

  /*
   * get the argument count and arguments sans the options
   */
  int getArgc() const;
  char *getArgv(int index) const;
  bool hasOptions() const;

private:                /* the hidden data structure */
  int argc;             /* command line arg count  */
  char **argv;          /* commnd line args */
  const char *filename; /* the option file */
  char *appname;        /* the application name from argv[0] */

  int *new_argv;      /* arguments sans options (index to argv) */
  int new_argc;       /* argument count sans the options */
  int max_legal_args; /* ignore extra arguments */

  /* option strings storage + indexing */
  int max_options;      /* maximum number of options */
  const char **options; /* storage */
  int *optiontype;      /* type - common, command, file */
  int *optionindex;     /* index into value storage */
  int option_counter;   /* counter for added options  */

  /* option chars storage + indexing */
  int max_char_options; /* maximum number options */
  char *optionchars;    /*  storage */
  int *optchartype;     /* type - common, command, file */
  int *optcharindex;    /* index into value storage */
  int optchar_counter;  /* counter for added options  */

  /* values */
  char **values;       /* common value storage */
  int g_value_counter; /* globally updated value index LAME! */

  /* help and usage */
  const char **usage;  /* usage */
  int max_usage_lines; /* max usage lines reserved */
  int usage_lines;     /* number of usage lines */

  bool command_set;   /* if argc/argv were provided */
  bool file_set;      /* if a filename was provided */
  bool mem_allocated; /* if memory allocated in init() */
  bool posix_style;   /* enables to turn off POSIX style options */
  bool verbose;       /* silent|verbose */
  bool print_usage;   /* usage verbose */
  bool print_help;    /* help verbose */

  char opt_prefix_char;                             /*  '-' in "-w" */
  char long_opt_prefix[MAX_LONG_PREFIX_LENGTH + 1]; /* '--' in "--width" */
  char file_delimiter_char;                         /* ':' in width : 100 */
  char file_comment_char; /*  '#' in "#this is a comment" */
  char equalsign;
  char comment;
  char delimiter;
  char endofline;
  char whitespace;
  char nullterminate;

  bool set;  // was static member
  bool once; // was static member

  bool hasoptions;
  bool autousage;

private: /* the hidden utils */
  void init();
  void init(int maxopt, int maxcharopt);
  bool alloc();
  void allocValues(int index, size_t length);
  void cleanup();
  bool valueStoreOK();

  /* grow storage arrays as required */
  bool doubleOptStorage();
  bool doubleCharStorage();
  bool doubleUsageStorage();

  bool setValue(const char *option, char *value);
  bool setFlagOn(const char *option);
  bool setValue(char optchar, char *value);
  bool setFlagOn(char optchar);

  void addOption(const char *option, int type);
  void addOption(char optchar, int type);
  void addOptionError(const char *opt) const;
  void addOptionError(char opt) const;
  bool findFlag(char *value);
  void addUsageError(const char *line);
  bool CommandSet() const;
  bool FileSet() const;
  bool POSIX() const;

  char parsePOSIX(char *arg);
  int parseGNU(char *arg);
  bool matchChar(char c);
  int matchOpt(char *opt);

  /* dot file methods */
  char *readFile();
  char *readFile(const char *fname);
  bool consumeFile(char *buffer);
  void processLine(char *theline, int length);
  char *chomp(char *str);
  void valuePairs(char *type, char *value);
  void justValue(char *value);

  void printVerbose(const char *msg) const;
  void printVerbose(char *msg) const;
  void printVerbose(char ch) const;
  void printVerbose() const;
};


AnyOption::AnyOption() { init(); }

AnyOption::AnyOption(int maxopt) { init(maxopt, maxopt); }

AnyOption::AnyOption(int maxopt, int maxcharopt) { init(maxopt, maxcharopt); }

AnyOption::~AnyOption() {
  if (mem_allocated)
    cleanup();
}

void AnyOption::init() { init(DEFAULT_MAXOPTS, DEFAULT_MAXOPTS); }

void AnyOption::init(int maxopt, int maxcharopt) {

  max_options = maxopt;
  max_char_options = maxcharopt;
  max_usage_lines = DEFAULT_MAXUSAGE;
  usage_lines = 0;
  argc = 0;
  argv = NULL;
  posix_style = true;
  verbose = false;
  filename = NULL;
  appname = NULL;
  option_counter = 0;
  optchar_counter = 0;
  new_argv = NULL;
  new_argc = 0;
  max_legal_args = 0;
  command_set = false;
  file_set = false;
  values = NULL;
  g_value_counter = 0;
  mem_allocated = false;
  opt_prefix_char = '-';
  file_delimiter_char = ':';
  file_comment_char = '#';
  equalsign = '=';
  comment = '#';
  delimiter = ':';
  endofline = '\n';
  whitespace = ' ';
  nullterminate = '\0';
  set = false;
  once = true;
  hasoptions = false;
  autousage = false;
  print_usage = false;
  print_help = false;

  strcpy(long_opt_prefix, "--");

  if (alloc() == false) {
    cout << endl << "OPTIONS ERROR : Failed allocating memory";
    cout << endl;
    cout << "Exiting." << endl;
    exit(0);
  }
}

bool AnyOption::alloc() {
  int i = 0;
  int size = 0;

  if (mem_allocated)
    return true;

  size = (max_options + 1) * sizeof(const char *);
  options = (const char **)malloc(size);
  optiontype = (int *)malloc((max_options + 1) * sizeof(int));
  optionindex = (int *)malloc((max_options + 1) * sizeof(int));
  if (options == NULL || optiontype == NULL || optionindex == NULL)
    return false;
  else
    mem_allocated = true;
  for (i = 0; i < max_options; i++) {
    options[i] = NULL;
    optiontype[i] = 0;
    optionindex[i] = -1;
  }
  optionchars = (char *)malloc((max_char_options + 1) * sizeof(char));
  optchartype = (int *)malloc((max_char_options + 1) * sizeof(int));
  optcharindex = (int *)malloc((max_char_options + 1) * sizeof(int));
  if (optionchars == NULL || optchartype == NULL || optcharindex == NULL) {
    mem_allocated = false;
    return false;
  }
  for (i = 0; i < max_char_options; i++) {
    optionchars[i] = '0';
    optchartype[i] = 0;
    optcharindex[i] = -1;
  }

  size = (max_usage_lines + 1) * sizeof(const char *);
  usage = (const char **)malloc(size);

  if (usage == NULL) {
    mem_allocated = false;
    return false;
  }
  for (i = 0; i < max_usage_lines; i++)
    usage[i] = NULL;

  return true;
}

void AnyOption::allocValues(int index, size_t length) {
  if (values[index] == NULL) {
    values[index] = (char *)malloc(length);
  } else {
    free(values[index]);
    values[index] = (char *)malloc(length);
  }
}

bool AnyOption::doubleOptStorage() {
  const char **options_saved = options;
  options = (const char **)realloc(options, ((2 * max_options) + 1) *
      sizeof(const char *));
  if (options == NULL) {
    free(options_saved);
    return false;
  }
  int *optiontype_saved = optiontype;
  optiontype =
      (int *)realloc(optiontype, ((2 * max_options) + 1) * sizeof(int));
  if (optiontype == NULL) {
    free(optiontype_saved);
    return false;
  }
  int *optionindex_saved = optionindex;
  optionindex =
      (int *)realloc(optionindex, ((2 * max_options) + 1) * sizeof(int));
  if (optionindex == NULL) {
    free(optionindex_saved);
    return false;
  }
  /* init new storage */
  for (int i = max_options; i < 2 * max_options; i++) {
    options[i] = NULL;
    optiontype[i] = 0;
    optionindex[i] = -1;
  }
  max_options = 2 * max_options;
  return true;
}

bool AnyOption::doubleCharStorage() {
  char *optionchars_saved = optionchars;
  optionchars =
      (char *)realloc(optionchars, ((2 * max_char_options) + 1) * sizeof(char));
  if (optionchars == NULL) {
    free(optionchars_saved);
    return false;
  }
  int *optchartype_saved = optchartype;
  optchartype =
      (int *)realloc(optchartype, ((2 * max_char_options) + 1) * sizeof(int));
  if (optchartype == NULL) {
    free(optchartype_saved);
    return false;
  }
  int *optcharindex_saved = optcharindex;
  optcharindex =
      (int *)realloc(optcharindex, ((2 * max_char_options) + 1) * sizeof(int));
  if (optcharindex == NULL) {
    free(optcharindex_saved);
    return false;
  }
  /* init new storage */
  for (int i = max_char_options; i < 2 * max_char_options; i++) {
    optionchars[i] = '0';
    optchartype[i] = 0;
    optcharindex[i] = -1;
  }
  max_char_options = 2 * max_char_options;
  return true;
}

bool AnyOption::doubleUsageStorage() {
  const char **usage_saved = usage;
  usage = (const char **)realloc(usage, ((2 * max_usage_lines) + 1) *
      sizeof(const char *));
  if (usage == NULL) {
    free(usage_saved);
    return false;
  }
  for (int i = max_usage_lines; i < 2 * max_usage_lines; i++)
    usage[i] = NULL;
  max_usage_lines = 2 * max_usage_lines;
  return true;
}

void AnyOption::cleanup() {
  free(options);
  free(optiontype);
  free(optionindex);
  free(optionchars);
  free(optchartype);
  free(optcharindex);
  free(usage);
  if (values != NULL) {
    for (int i = 0; i < g_value_counter; i++) {
      free(values[i]);
      values[i] = NULL;
    }
    free(values);
  }
  if (new_argv != NULL)
    free(new_argv);
}

void AnyOption::setCommandPrefixChar(char _prefix) {
  opt_prefix_char = _prefix;
}

void AnyOption::setCommandLongPrefix(const char *_prefix) {
  if (strlen(_prefix) > MAX_LONG_PREFIX_LENGTH) {
    strncpy(long_opt_prefix, _prefix, MAX_LONG_PREFIX_LENGTH);
    long_opt_prefix[MAX_LONG_PREFIX_LENGTH] = nullterminate;
  } else {
    strcpy(long_opt_prefix, _prefix);
  }
}

void AnyOption::setFileCommentChar(char _comment) {
  file_delimiter_char = _comment;
}

void AnyOption::setFileDelimiterChar(char _delimiter) {
  file_comment_char = _delimiter;
}

bool AnyOption::CommandSet() const { return (command_set); }

bool AnyOption::FileSet() const { return (file_set); }

void AnyOption::noPOSIX() { posix_style = false; }

bool AnyOption::POSIX() const { return posix_style; }

void AnyOption::setVerbose() { verbose = true; }

void AnyOption::printVerbose() const {
  if (verbose)
    cout << endl;
}
void AnyOption::printVerbose(const char *msg) const {
  if (verbose)
    cout << msg;
}

void AnyOption::printVerbose(char *msg) const {
  if (verbose)
    cout << msg;
}

void AnyOption::printVerbose(char ch) const {
  if (verbose)
    cout << ch;
}

bool AnyOption::hasOptions() const { return hasoptions; }

void AnyOption::autoUsagePrint(bool _autousage) { autousage = _autousage; }

void AnyOption::useCommandArgs(int _argc, char **_argv) {
  argc = _argc;
  argv = _argv;
  command_set = true;
  appname = argv[0];
  if (argc > 1)
    hasoptions = true;
}

void AnyOption::useFiileName(const char *_filename) {
  filename = _filename;
  file_set = true;
}

/*
 * set methods for options
 */

void AnyOption::setCommandOption(const char *opt) {
  addOption(opt, COMMAND_OPT);
  g_value_counter++;
}

void AnyOption::setCommandOption(char opt) {
  addOption(opt, COMMAND_OPT);
  g_value_counter++;
}

void AnyOption::setCommandOption(const char *opt, char optchar) {
  addOption(opt, COMMAND_OPT);
  addOption(optchar, COMMAND_OPT);
  g_value_counter++;
}

void AnyOption::setCommandFlag(const char *opt) {
  addOption(opt, COMMAND_FLAG);
  g_value_counter++;
}

void AnyOption::setCommandFlag(char opt) {
  addOption(opt, COMMAND_FLAG);
  g_value_counter++;
}

void AnyOption::setCommandFlag(const char *opt, char optchar) {
  addOption(opt, COMMAND_FLAG);
  addOption(optchar, COMMAND_FLAG);
  g_value_counter++;
}

void AnyOption::setFileOption(const char *opt) {
  addOption(opt, FILE_OPT);
  g_value_counter++;
}

void AnyOption::setFileOption(char opt) {
  addOption(opt, FILE_OPT);
  g_value_counter++;
}

void AnyOption::setFileOption(const char *opt, char optchar) {
  addOption(opt, FILE_OPT);
  addOption(optchar, FILE_OPT);
  g_value_counter++;
}

void AnyOption::setFileFlag(const char *opt) {
  addOption(opt, FILE_FLAG);
  g_value_counter++;
}

void AnyOption::setFileFlag(char opt) {
  addOption(opt, FILE_FLAG);
  g_value_counter++;
}

void AnyOption::setFileFlag(const char *opt, char optchar) {
  addOption(opt, FILE_FLAG);
  addOption(optchar, FILE_FLAG);
  g_value_counter++;
}

void AnyOption::setOption(const char *opt) {
  addOption(opt, COMMON_OPT);
  g_value_counter++;
}

void AnyOption::setOption(char opt) {
  addOption(opt, COMMON_OPT);
  g_value_counter++;
}

void AnyOption::setOption(const char *opt, char optchar) {
  addOption(opt, COMMON_OPT);
  addOption(optchar, COMMON_OPT);
  g_value_counter++;
}

void AnyOption::setFlag(const char *opt) {
  addOption(opt, COMMON_FLAG);
  g_value_counter++;
}

void AnyOption::setFlag(const char opt) {
  addOption(opt, COMMON_FLAG);
  g_value_counter++;
}

void AnyOption::setFlag(const char *opt, char optchar) {
  addOption(opt, COMMON_FLAG);
  addOption(optchar, COMMON_FLAG);
  g_value_counter++;
}

void AnyOption::addOption(const char *opt, int type) {
  if (option_counter >= max_options) {
    if (doubleOptStorage() == false) {
      addOptionError(opt);
      return;
    }
  }
  options[option_counter] = opt;
  optiontype[option_counter] = type;
  optionindex[option_counter] = g_value_counter;
  option_counter++;
}

void AnyOption::addOption(char opt, int type) {
  if (!POSIX()) {
    printVerbose("Ignoring the option character \"");
    printVerbose(opt);
    printVerbose("\" ( POSIX options are turned off )");
    printVerbose();
    return;
  }

  if (optchar_counter >= max_char_options) {
    if (doubleCharStorage() == false) {
      addOptionError(opt);
      return;
    }
  }
  optionchars[optchar_counter] = opt;
  optchartype[optchar_counter] = type;
  optcharindex[optchar_counter] = g_value_counter;
  optchar_counter++;
}

void AnyOption::addOptionError(const char *opt) const {
  cout << endl;
  cout << "OPTIONS ERROR : Failed allocating extra memory " << endl;
  cout << "While adding the option : \"" << opt << "\"" << endl;
  cout << "Exiting." << endl;
  cout << endl;
  exit(0);
}

void AnyOption::addOptionError(char opt) const {
  cout << endl;
  cout << "OPTIONS ERROR : Failed allocating extra memory " << endl;
  cout << "While adding the option: \"" << opt << "\"" << endl;
  cout << "Exiting." << endl;
  cout << endl;
  exit(0);
}

void AnyOption::processOptions() {
  if (!valueStoreOK())
    return;
}

void AnyOption::processCommandArgs(int max_args) {
  max_legal_args = max_args;
  processCommandArgs();
}

void AnyOption::processCommandArgs(int _argc, char **_argv, int max_args) {
  max_legal_args = max_args;
  processCommandArgs(_argc, _argv);
}

void AnyOption::processCommandArgs(int _argc, char **_argv) {
  useCommandArgs(_argc, _argv);
  processCommandArgs();
}

void AnyOption::processCommandArgs() {
  if (!(valueStoreOK() && CommandSet()))
    return;

  if (max_legal_args == 0)
    max_legal_args = argc;
  new_argv = (int *)malloc((max_legal_args + 1) * sizeof(int));
  for (int i = 1; i < argc; i++) { /* ignore first argv */
    if (argv[i][0] == long_opt_prefix[0] &&
        argv[i][1] == long_opt_prefix[1]) { /* long GNU option */
      int match_at = parseGNU(argv[i] + 2); /* skip -- */
      if (match_at >= 0 && i < argc - 1)    /* found match */
        setValue(options[match_at], argv[++i]);
    } else if (argv[i][0] == opt_prefix_char) { /* POSIX char */
      if (POSIX()) {
        char ch = parsePOSIX(argv[i] + 1); /* skip - */
        if (ch != '0' && i < argc - 1)     /* matching char */
          setValue(ch, argv[++i]);
      } else { /* treat it as GNU option with a - */
        int match_at = parseGNU(argv[i] + 1); /* skip - */
        if (match_at >= 0 && i < argc - 1)    /* found match */
          setValue(options[match_at], argv[++i]);
      }
    } else { /* not option but an argument keep index */
      if (new_argc < max_legal_args) {
        new_argv[new_argc] = i;
        new_argc++;
      } else { /* ignore extra arguments */
        printVerbose("Ignoring extra argument: ");
        printVerbose(argv[i]);
        printVerbose();
        printAutoUsage();
      }
      printVerbose("Unknown command argument option : ");
      printVerbose(argv[i]);
      printVerbose();
      printAutoUsage();
    }
  }
}

char AnyOption::parsePOSIX(char *arg) {

  for (unsigned int i = 0; i < strlen(arg); i++) {
    char ch = arg[i];
    if (matchChar(ch)) { /* keep matching flags till an option */
      /*if last char argv[++i] is the value */
      if (i == strlen(arg) - 1) {
        return ch;
      } else { /* else the rest of arg is the value */
        i++;   /* skip any '=' and ' ' */
        while (arg[i] == whitespace || arg[i] == equalsign)
          i++;
        setValue(ch, arg + i);
        return '0';
      }
    }
  }
  printVerbose("Unknown command argument option : ");
  printVerbose(arg);
  printVerbose();
  printAutoUsage();
  return '0';
}

int AnyOption::parseGNU(char *arg) {
  size_t split_at = 0;
  /* if has a '=' sign get value */
  for (size_t i = 0; i < strlen(arg); i++) {
    if (arg[i] == equalsign) {
      split_at = i;    /* store index */
      i = strlen(arg); /* get out of loop */
    }
  }
  if (split_at > 0) { /* it is an option value pair */
    char *tmp = (char *)malloc((split_at + 1) * sizeof(char));
    for (size_t i = 0; i < split_at; i++)
      tmp[i] = arg[i];
    tmp[split_at] = '\0';

    if (matchOpt(tmp) >= 0) {
      setValue(options[matchOpt(tmp)], arg + split_at + 1);
      free(tmp);
    } else {
      printVerbose("Unknown command argument option : ");
      printVerbose(arg);
      printVerbose();
      printAutoUsage();
      free(tmp);
      return -1;
    }
  } else { /* regular options with no '=' sign  */
    return matchOpt(arg);
  }
  return -1;
}

int AnyOption::matchOpt(char *opt) {
  for (int i = 0; i < option_counter; i++) {
    if (strcmp(options[i], opt) == 0) {
      if (optiontype[i] == COMMON_OPT ||
          optiontype[i] == COMMAND_OPT) { /* found option return index */
        return i;
      } else if (optiontype[i] == COMMON_FLAG ||
          optiontype[i] == COMMAND_FLAG) { /* found flag, set it */
        setFlagOn(opt);
        return -1;
      }
    }
  }
  printVerbose("Unknown command argument option : ");
  printVerbose(opt);
  printVerbose();
  printAutoUsage();
  return -1;
}
bool AnyOption::matchChar(char c) {
  for (int i = 0; i < optchar_counter; i++) {
    if (optionchars[i] == c) { /* found match */
      if (optchartype[i] == COMMON_OPT ||
          optchartype[i] ==
              COMMAND_OPT) { /* an option store and stop scanning */
        return true;
      } else if (optchartype[i] == COMMON_FLAG ||
          optchartype[i] ==
              COMMAND_FLAG) { /* a flag store and keep scanning */
        setFlagOn(c);
        return false;
      }
    }
  }
  printVerbose("Unknown command argument option : ");
  printVerbose(c);
  printVerbose();
  printAutoUsage();
  return false;
}

bool AnyOption::valueStoreOK() {
  if (!set) {
    if (g_value_counter > 0) {
      const int size = g_value_counter * sizeof(char *);
      values = (char **)malloc(size);
      for (int i = 0; i < g_value_counter; i++)
        values[i] = NULL;
      set = true;
    }
  }
  return set;
}

/*
 * public get methods
 */
char *AnyOption::getValue(const char *option) {
  if (!valueStoreOK())
    return NULL;

  for (int i = 0; i < option_counter; i++) {
    if (strcmp(options[i], option) == 0)
      return values[optionindex[i]];
  }
  return NULL;
}

bool AnyOption::getFlag(const char *option) {
  if (!valueStoreOK())
    return false;
  for (int i = 0; i < option_counter; i++) {
    if (strcmp(options[i], option) == 0)
      return findFlag(values[optionindex[i]]);
  }
  return false;
}

char *AnyOption::getValue(char option) {
  if (!valueStoreOK())
    return NULL;
  for (int i = 0; i < optchar_counter; i++) {
    if (optionchars[i] == option)
      return values[optcharindex[i]];
  }
  return NULL;
}

bool AnyOption::getFlag(char option) {
  if (!valueStoreOK())
    return false;
  for (int i = 0; i < optchar_counter; i++) {
    if (optionchars[i] == option)
      return findFlag(values[optcharindex[i]]);
  }
  return false;
}

bool AnyOption::findFlag(char *val) {
  if (val == NULL)
    return false;

  if (strcmp(TRUE_FLAG, val) == 0)
    return true;

  return false;
}

/*
 * private set methods
 */
bool AnyOption::setValue(const char *option, char *value) {
  if (!valueStoreOK())
    return false;
  for (int i = 0; i < option_counter; i++) {
    if (strcmp(options[i], option) == 0) {
      size_t length = (strlen(value) + 1) * sizeof(char);
      allocValues(optionindex[i], length);
      strncpy(values[optionindex[i]], value, length);
      return true;
    }
  }
  return false;
}

bool AnyOption::setFlagOn(const char *option) {
  if (!valueStoreOK())
    return false;
  for (int i = 0; i < option_counter; i++) {
    if (strcmp(options[i], option) == 0) {
      size_t length = (strlen(TRUE_FLAG) + 1) * sizeof(char);
      allocValues(optionindex[i], length);
      strncpy(values[optionindex[i]], TRUE_FLAG, length);
      return true;
    }
  }
  return false;
}

bool AnyOption::setValue(char option, char *value) {
  if (!valueStoreOK())
    return false;
  for (int i = 0; i < optchar_counter; i++) {
    if (optionchars[i] == option) {
      size_t length = (strlen(value) + 1) * sizeof(char);
      allocValues(optcharindex[i], length);
      strncpy(values[optcharindex[i]], value, length);
      return true;
    }
  }
  return false;
}

bool AnyOption::setFlagOn(char option) {
  if (!valueStoreOK())
    return false;
  for (int i = 0; i < optchar_counter; i++) {
    if (optionchars[i] == option) {
      size_t length = (strlen(TRUE_FLAG) + 1) * sizeof(char);
      allocValues(optcharindex[i], length);
      strncpy(values[optcharindex[i]], TRUE_FLAG, length);
      return true;
    }
  }
  return false;
}

int AnyOption::getArgc() const { return new_argc; }

char *AnyOption::getArgv(int index) const {
  if (index < new_argc) {
    return (argv[new_argv[index]]);
  }
  return NULL;
}

/* option file sub routines */

bool AnyOption::processFile() {
  if (!(valueStoreOK() && FileSet()))
    return false;
  return hasoptions = (consumeFile(readFile()));
}

bool AnyOption::processFile(const char *_filename) {
  useFiileName(_filename);
  return (processFile());
}

char *AnyOption::readFile() { return (readFile(filename)); }

/*
 * read the file contents to a character buffer
 */

char *AnyOption::readFile(const char *fname) {
  char *buffer;
  ifstream is;
  is.open(fname, ifstream::in);
  if (!is.good()) {
    is.close();
    return NULL;
  }
  is.seekg(0, ios::end);
  size_t length = (size_t)is.tellg();
  is.seekg(0, ios::beg);
  buffer = (char *)malloc((length + 1) * sizeof(char));
  is.read(buffer, length);
  is.close();
  buffer[length] = nullterminate;
  return buffer;
}

/*
 * scans a char* buffer for lines that does not
 * start with the specified comment character.
 */
bool AnyOption::consumeFile(char *buffer) {

  if (buffer == NULL)
    return false;

  char *cursor = buffer; /* preserve the ptr */
  char *pline = NULL;
  int linelength = 0;
  bool newline = true;
  for (unsigned int i = 0; i < strlen(buffer); i++) {
    if (*cursor == endofline) { /* end of line */
      if (pline != NULL)        /* valid line */
        processLine(pline, linelength);
      pline = NULL;
      newline = true;
    } else if (newline) { /* start of line */
      newline = false;
      if ((*cursor != comment)) { /* not a comment */
        pline = cursor;
        linelength = 0;
      }
    }
    cursor++; /* keep moving */
    linelength++;
  }
  free(buffer);
  return true;
}

/*
 *  find a valid type value pair separated by a delimiter
 *  character and pass it to valuePairs()
 *  any line which is not valid will be considered a value
 *  and will get passed on to justValue()
 *
 *  assuming delimiter is ':' the behaviour will be,
 *
 *  width:10    - valid pair valuePairs( width, 10 );
 *  width : 10  - valid pair valuepairs( width, 10 );
 *
 *  ::::        - not valid
 *  width       - not valid
 *  :10         - not valid
 *  width:      - not valid
 *  ::          - not valid
 *  :           - not valid
 *
 */

void AnyOption::processLine(char *theline, int length) {
  char *pline = (char *)malloc((length + 1) * sizeof(char));
  for (int i = 0; i < length; i++)
    pline[i] = *(theline++);
  pline[length] = nullterminate;
  char *cursor = pline; /* preserve the ptr */
  if (*cursor == delimiter || *(cursor + length - 1) == delimiter) {
    justValue(pline); /* line with start/end delimiter */
  } else {
    bool found = false;
    for (int i = 1; i < length - 1 && !found; i++) { /* delimiter */
      if (*cursor == delimiter) {
        *(cursor - 1) = nullterminate; /* two strings */
        found = true;
        valuePairs(pline, cursor + 1);
      }
      cursor++;
    }
    cursor++;
    if (!found) /* not a pair */
      justValue(pline);
  }
  free(pline);
}

/*
 * removes trailing and preceding white spaces from a string
 */
char *AnyOption::chomp(char *str) {
  while (*str == whitespace)
    str++;
  char *end = str + strlen(str) - 1;
  while (*end == whitespace)
    end--;
  *(end + 1) = nullterminate;
  return str;
}

void AnyOption::valuePairs(char *type, char *value) {
  if (strlen(chomp(type)) == 1) { /* this is a char option */
    for (int i = 0; i < optchar_counter; i++) {
      if (optionchars[i] == type[0]) { /* match */
        if (optchartype[i] == COMMON_OPT || optchartype[i] == FILE_OPT) {
          setValue(type[0], chomp(value));
          return;
        }
      }
    }
  }
  /* if no char options matched */
  for (int i = 0; i < option_counter; i++) {
    if (strcmp(options[i], type) == 0) { /* match */
      if (optiontype[i] == COMMON_OPT || optiontype[i] == FILE_OPT) {
        setValue(type, chomp(value));
        return;
      }
    }
  }
  printVerbose("Unknown option in resource file : ");
  printVerbose(type);
  printVerbose();
}

void AnyOption::justValue(char *type) {

  if (strlen(chomp(type)) == 1) { /* this is a char option */
    for (int i = 0; i < optchar_counter; i++) {
      if (optionchars[i] == type[0]) { /* match */
        if (optchartype[i] == COMMON_FLAG || optchartype[i] == FILE_FLAG) {
          setFlagOn(type[0]);
          return;
        }
      }
    }
  }
  /* if no char options matched */
  for (int i = 0; i < option_counter; i++) {
    if (strcmp(options[i], type) == 0) { /* match */
      if (optiontype[i] == COMMON_FLAG || optiontype[i] == FILE_FLAG) {
        setFlagOn(type);
        return;
      }
    }
  }
  printVerbose("Unknown option in resource file : ");
  printVerbose(type);
  printVerbose();
}

/*
 * usage and help
 */

void AnyOption::printAutoUsage() {
  if (autousage)
    printUsage();
}

void AnyOption::printUsage() {

  if (once) {
    once = false;
    cout << endl;
    for (int i = 0; i < usage_lines; i++)
      cout << usage[i] << endl;
    cout << endl;
  }
}

void AnyOption::addUsage(const char *line) {
  if (usage_lines >= max_usage_lines) {
    if (doubleUsageStorage() == false) {
      addUsageError(line);
      exit(1);
    }
  }
  usage[usage_lines] = line;
  usage_lines++;
}

void AnyOption::addUsageError(const char *line) {
  cout << endl;
  cout << "OPTIONS ERROR : Failed allocating extra memory " << endl;
  cout << "While adding the usage/help  : \"" << line << "\"" << endl;
  cout << "Exiting." << endl;
  cout << endl;
  exit(0);
}

