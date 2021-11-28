include "ossp_base.mzn";
  
solve :: seq_search([int_search([job_task_start[j, k] | j in jobs, k in tasks], smallest, indomain_min, complete),
		                   int_search([objective], input_order, indomain_min, complete)])
  minimize objective;
