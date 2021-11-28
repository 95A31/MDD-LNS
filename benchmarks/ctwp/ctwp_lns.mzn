include "restart.mzn";
include "ctwp_base.mzn";

predicate ctwp_random_nbh() =
  if status() != START then
    forall (i in Cavities) (if uniform_internal(1,100) < 70 then cfp[i] = sol(cfp[i]) endif)
  endif;

solve ::on_restart("ctwp_random_nbh") minimize obj;
