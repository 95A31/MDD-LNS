include "sop_base.mzn";

solve ::int_search(path, first_fail, indomain_min, complete) minimize total_distance;
