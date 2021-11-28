include "restart.mzn";
include "sop_base.mzn";

predicate sop_random_nbh() =
  if status() != START then
    forall (i in 1..n) (if uniform_internal(1,100) < 70 then path[i] = sol(path[i]) endif)
  endif;

solve ::on_restart("sop_random_nbh") ::int_search(path, first_fail, indomain_min, complete)  minimize total_distance;