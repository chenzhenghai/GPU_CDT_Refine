# GPU_CDT_Refine
A 2D CDT Refinement Software on GPU

* A NVIDIA GPU is required since this project is implemented using CUDA

Refinement Routine:
void GPU_Refine_Quality(
  triangulateio *input, 
  triangulateio *result, 
  double theta, 
  InsertPolicy insertpolicy,
  DeletePolicy deletepolicy, 
  int mode,
  int debug_iter, 
  PStatus **ps_debug,
  TStatus **ts_debug)

triangulateio *input:
  
