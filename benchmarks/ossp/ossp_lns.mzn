include "restart.mzn";
include "ossp_base.mzn";

predicate osp_random_nbh() =
  if status() != START then
    forall (j in jobs, k in tasks) (if uniform_internal(1,100) < 70 then job_task_start[j,k] = sol(job_task_start[j,k]) endif)
  endif;
  
solve ::on_restart("osp_random_nbh") 
	:: seq_search([
		int_search([job_task_start[j, k] | j in jobs, k in tasks], 
			smallest, indomain_min, complete),
		int_search([objective], input_order, indomain_min, complete)
	])
	minimize objective;