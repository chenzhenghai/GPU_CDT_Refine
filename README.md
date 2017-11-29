# GPU_CDT_Refine
A 2D CDT Refinement Software on GPU

* A NVIDIA GPU is required since this project is implemented using CUDA

Refinement Routine (located in refine.h and refine.cu):<br/>
void GPU_Refine_Quality(  
&nbsp;&nbsp;&nbsp;&nbsp; triangulateio *input,  
&nbsp;&nbsp;&nbsp;&nbsp; triangulateio *result,  
&nbsp;&nbsp;&nbsp;&nbsp; double theta,  
&nbsp;&nbsp;&nbsp;&nbsp; InsertPolicy insertpolicy,  
&nbsp;&nbsp;&nbsp;&nbsp; DeletePolicy deletepolicy,  
&nbsp;&nbsp;&nbsp;&nbsp; int mode,  
&nbsp;&nbsp;&nbsp;&nbsp; int debug_iter,  
&nbsp;&nbsp;&nbsp;&nbsp; PStatus **ps_debug,  
&nbsp;&nbsp;&nbsp;&nbsp; TStatus **ts_debug)  
 
triangulateio *input:  
Input is a constrained Delaunay triangulation. Use the same data type "triangulateio" as Triangle software (https://www.cs.cmu.edu/~quake/triangle.html), see triangle.h and triangle.cpp for more information. GenerateRandomInput routine in mesh.h and mesh.cpp is able to generate a random point set and a random segment set, then compute the CDT of them as an input. You can make your own input as well. Some input samples are also give in input folder.

triangulateio *result:  
A pointer. Make sure it is not NULL. Its pointlist, trianglelist and segmentlist contain the final refined CDT.

double theta:  
Minimum allowable angle. Theoretically, it cannot be smaller than 20.7 degree. The triangle in final mesh wouldn't contain angle smaller than theta.

InsertPolicy insertpolicy:  
Enumeration type defined in refine.h. It has two possible values: Circumcenter and Offcenter, which correspond to two different types of center to be inserted.

DeletePolicy deletepolicy:  
Enumeration type defined in refine.h. It has not yet been used.

int mode:  
Refinement mode. When mode is 1, Ruppert's mode is used, otherwise, Chew's.

int debug_iter:  
Debug iteration number. For debug only.

PStatus **ps_debug:  
The debug pointer of the status of point. For debug only.

TStatus **ts_debug:
The debug pointer of the status of triangle. For debug only.

Proceed to main.cpp to check how to call gpu refinement routine properly.

