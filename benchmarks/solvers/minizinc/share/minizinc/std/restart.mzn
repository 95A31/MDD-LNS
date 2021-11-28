/***
	@groupdef internal Restart Builtins

	These annotations and functions provide the basic building blocks to implement
	meta-heuristics and governed search with the use of solver internal values and
	functionality.
*/

/*
	A search annotation that supplies a function to be executed on restart
*/
ann: on_restart(string: pred);

/*
	The 'status' function reports the status of the solver (before restarting).
*/
enum STATUS = {START, UNKNOWN, UNSAT, SAT, OPT};
function var STATUS: status();

/*
	When complete is set to 'true', then it marks the search as complete.
*/
predicate complete();% = abort("'complete' should be used in reified context");
predicate complete_reif(var bool: marker);

/*
	The 'uniform' function provides random values chosen by the solver. The arguments
	to the function limit the values to a certain domain.
*/
function var int: uniform_internal(int: i, int: j) ::impure = int_uniform(i, j);

function var int: uniform_internal(set of int: S) ::impure =
	if card(S) == max(S) - min(S) + 1 then
		int_uniform(min(S),max(S))
	else
		[ i | i in S ][int_uniform(1,card(S))]
	endif;

/*
	'int_uniform' is the random intrinsic that needs to be implemented by the solver.
*/
function var int: int_uniform(int: a, int: b) ::impure;

/*
	The 'lastval' functions provides access to last values taken by a model
	variables. The lastval functions are only safe to use when the STATUS has been
	SAT or OPT.
*/
function var bool: lastval(var bool: x) =
	if is_fixed(x) then
		fix(x)
	else
		bool_lastval(x)
	endif;

function var float: lastval(var float: x) =
	if is_fixed(x) then
		fix(x)
	else
		float_lastval(x)
	endif;

function var int: lastval(var int: x) =
	if is_fixed(x) then
		fix(x)
	else
		int_lastval(x)
	endif;

/*
	'bool_lastval', 'float_lastval', and 'int_lastval' are the solution intrinsics that needs
	to be implemented by the solver.
*/
function var bool: bool_lastval(var bool: a);
function var float: float_lastval(var float: a);
function var int: int_lastval(var int: a);

/*
	The 'sol' functions provides access to solution values of model variables. The
	lastval functions are only safe to use when the STATUS has been SAT or OPT.
*/
function var bool: sol(var bool: x) =
	if is_fixed(x) then
		fix(x)
	else
		bool_sol(x)
	endif;

function var float: sol(var float: x) =
	if is_fixed(x) then
		fix(x)
	else
		float_sol(x)
	endif;

function var int: sol(var int: x) =
	if is_fixed(x) then
		fix(x)
	else
		int_sol(x)
	endif;

function array[int] of var bool: sol(array[int] of var bool: x) =
	[if is_fixed(x[i]) then fix(x[i]) else bool_sol(x[i]) endif | i in index_set(x)];

function array[int] of var float: sol(array[int] of var float: x) =
	[if is_fixed(x[i]) then fix(x[i]) else float_sol(x[i]) endif | i in index_set(x)];

function array[int] of var int: sol(array[int] of var int: x) =
	[if is_fixed(x[i]) then fix(x[i]) else int_sol(x[i]) endif | i in index_set(x)];

/*
	'bool_rnd', 'float_sol', and 'int_sol' are the solution intrinsics that needs
	to be implemented by the solver.
*/
function var bool: bool_sol(var bool: a);
function var float: float_sol(var float: a);
function var int: int_sol(var int: a);

/*
	'basic_lns' provides LNS in its simplest form where one nbh is applied on
	every restart.
*/
predicate basic_lns(var bool: nbh) =
    (status() != START) -> nbh;

/*
	'round_robin' provides a metaheuristic for LNS where each neighbourhood is
	chosen sequentially.
*/
predicate round_robin(array[int] of var bool: nbhs) =
	let {
		int: len = length(nbhs);
		% Transposed neighbourhoods in case of index set not starting from 1
		array[1..len] of var bool: t_nbhs = [ bv | bv in nbhs];
		% Neighbourhood selection
		var -1..len-1: select;
	} in forall(i in 1..len) ( (select-1 == i) -> t_nbhs[i]) /\
	    if status() == START then
			select = -1
		else
		    select = (sol(select) + 1) mod len
		endif;
