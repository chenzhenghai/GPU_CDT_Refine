#include <stdio.h>
#include <cuda_runtime.h>
#include <windows.h>
#include <time.h>
#include "refine.h"
#include "mesh.h"
#include "predicates.h"
#include "freeglut\freeglut.h"

//************************************
// Debug utilities

triangulateio *draw_result;
int draw_numofpoints;
PStatus * debug_ps = NULL;
TStatus * debug_ts = NULL;

// Find orientataion of tri2's edge that is incident to tri1
int findIncidentOri_Host(int * trianglelist, int tri1, int tri2)
{
	if( tri1 < 0 || tri2 < 0 )
		return -1;
	int inc0=-1,inc1=-1;
	int tri1_p[3] = {
		trianglelist[3*tri1],
		trianglelist[3*tri1+1],
		trianglelist[3*tri1+2]
	};
	int tri2_p[3] = {
		trianglelist[3*tri2],
		trianglelist[3*tri2+1],
		trianglelist[3*tri2+2]
	};

	// incident edge
	int count = 0;
	for(int i=0; i<3; i++)
	{
		for(int j=0; j<3; j++)
		{
			if( tri1_p[i] == tri2_p[j] )
			{
				if (count == 0)
				{
					inc0 = j;
					count += 1;
					continue;
				}
				else
				{
					inc1 = j;
					break;
				}
			}
		}
	}
	
	if(inc0 == -1 || inc1 == -1) // not found
		return -1;

	// orientation
	int differ = inc0 - inc1;
	int index;
	if ( differ == -1 || differ == 2 )
		index =  inc0;
	else
		index =  inc1;

	if(index==0)
		return 2;
	else if(index==1)
		return 0;
	else if(index==2)
		return 1;
	else
		return -1;
}

bool checkNeighbors(triangulateio *input)
{
	for(int i=0; i<input->numberoftriangles; i++)
	{
		for(int j=0; j<3; j++)
		{
			int neighbor = input->neighborlist[3*i+j];
			if(neighbor == -1)
				continue;
			int flag = 0;
			for(int k=0; k<3;k++)
			{
				if(input->neighborlist[3*neighbor+k] == i)
				{
					flag++;
					if(findIncidentOri_Host(input->trianglelist,i,neighbor) != -1)
						flag++;
					break;
				}
			}
			if(flag != 2)
			{
				printf("Invalid vertices or neighbors %d: (%d,%d,%d) - (%d,%d,%d) | %d: (%d,%d,%d) - (%d,%d,%d) !\n",
					i, 
					input->trianglelist[3*i],input->trianglelist[3*i+1],input->trianglelist[3*i+2],
					input->neighborlist[3*i],input->neighborlist[3*i+1],input->neighborlist[3*i+2],	
					neighbor,
					input->trianglelist[3*neighbor],input->trianglelist[3*neighbor+1],input->trianglelist[3*neighbor+2],
					input->neighborlist[3*neighbor],input->neighborlist[3*neighbor+1],input->neighborlist[3*neighbor+2]);
				return false;
			}
		}
	}
	return true;
}

void printTriangles(triangulateio *input)
{
	for(int i=0; i<input->numberoftriangles; i++)
	{
		printf("%d: ",i);
		for(int j=0; j<3; j++)
			printf("%d ",input->trianglelist[3*i+j]);
		printf("- ");
		for(int j=0; j<3; j++)
			printf("%d ",input->neighborlist[3*i+j]);
		printf("\n");
	}
}

void printPoints(triangulateio *input)
{
	for(int i=0; i<input->numberofpoints; i++)
	{
		printf("%f, %f",input->pointlist[2*i],input->pointlist[2*i+1]);
		if(debug_ps !=NULL)
		{
			if(debug_ps[i].isSegmentSplit())
				printf(" midpoint");
			else
				printf(" steiner");
		}
		printf("\n");
	}
}

void printSegments(triangulateio *input)
{
	for(int i=0; i<input->numberofsegments; i++)
	{
		printf("%d,%d\n",input->segmentlist[2*i],input->segmentlist[2*i+1]);
	}
}

bool checkIncircle(triangulateio *input)
{
	int edgecount = 0;
	int segcount = 0;

	for(int i=0; i<input->numberoftriangles; i++)
	{
		for(int ori=0; ori<3; ori++)
		{
			int neighbor = input->neighborlist[3*i+ori];
			if(neighbor == -1)
			{
				edgecount += 2;
				segcount +=2;
				continue;
			}
			else
				edgecount++;

			int org, dest, apex;
			double x,y;
			
			// origin point
			org = input->trianglelist[3*i + (ori+1)%3];			
			x = input->pointlist[2*org];
			y = input->pointlist[2*org+1];
			triVertex triOrg(x,y);

			// destination point
			dest = input->trianglelist[3*i + (ori+2)%3];
			x = input->pointlist[2*dest];
			y = input->pointlist[2*dest+1];
			triVertex triDest(x,y);

			bool seg = false;
			for(int j=0; j<input->numberofsegments; j++)
			{
				int p0 = input->segmentlist[2*j];
				int p1 = input->segmentlist[2*j+1];
				if( (org == p0 && dest == p1) || (org == p1 && dest == p0))
				{
					// dont need to check this edge
					segcount++;
					seg = true;
					break;
				}
			}
			if(seg)
				continue; // skip this edge

			// apex point
			apex = input->trianglelist[3*i + ori];
			x = input->pointlist[2*apex];
			y = input->pointlist[2*apex+1];
			triVertex triApex(x,y);

			// opposite Apex point
			int oppOri = findIncidentOri_Host(input->trianglelist,i,neighbor);
			int oppApex = input->trianglelist[3*neighbor+oppOri];
			x = input->pointlist[2*oppApex];
			y = input->pointlist[2*oppApex+1];
			triVertex triOppApex(x,y);

			REAL test = incircle(&triOrg,&triDest,&triApex,&triOppApex);

			if(test > 0)
			{
				printf("Incircle test fail for triangle %d and %d - %d(%f,%f), %d(%f,%f), %d(%f,%f), %d(%f,%f)",
					i,neighbor,
					org,input->pointlist[2*org],input->pointlist[2*org+1],
					dest,input->pointlist[2*dest],input->pointlist[2*dest+1],
					apex,input->pointlist[2*apex],input->pointlist[2*apex+1],
					oppApex,input->pointlist[2*oppApex],input->pointlist[2*oppApex+1]);
				printf(",incircle = %lf\n",test);
				//return false;
			}

		}
	}
	//printf("%d\n",segcount/2);
	edgecount /= 2;
	int euler = input->numberofpoints - edgecount + input->numberoftriangles + 1;
	if(euler != 2)
	{
		printf("Euler equation test fail!\n");
		return false;
	}

	return true;
}

bool isPureBadTriangle(triVertex vOrg, triVertex vDest, triVertex vApex, double theta)
{
	REAL dx[3], dy[3], edgelength[3];

	REAL goodAngle = cos(theta * PI/180.0);	
	goodAngle *= goodAngle;

	triVertex p[3];
	p[0] = vOrg;
	p[1] = vDest;
	p[2] = vApex;

	for (int i = 0; i < 3; i++) 
	{
		int j = (i + 1) % 3;
		int k = (i + 2) % 3;
		dx[i] = p[j].x - p[k].x;
		dy[i] = p[j].y - p[k].y;
		edgelength[i] = dx[i] * dx[i] + dy[i] * dy[i];			 
	}

	for (int i = 0; i < 3; i++) 
	{
		int  j = (i + 1) % 3;
		int  k = (i + 2) % 3;
		REAL dotproduct = dx[j] * dx[k] + dy[j] * dy[k];
		REAL cossquare = dotproduct * dotproduct / (edgelength[j] * edgelength[k]);
		if(cossquare > goodAngle)
		{
			return true;
		}
	}

	return false;
}

bool checkQuality(triangulateio *input, double theta)
{
	triVertex p[3];
	REAL dx[3], dy[3], edgelength[3];

	REAL goodAngle = cos(theta * PI/180.0);	
	goodAngle *= goodAngle;

	for(int num=0; num < input->numberoftriangles; num++)
	{
		int org,dest,apex;

		org = input->trianglelist[3*num+1];
		dest = input->trianglelist[3*num+2];
		apex = input->trianglelist[3*num];

		p[0].x = input->pointlist[2*org];
		p[0].y = input->pointlist[2*org+1];
		p[1].x = input->pointlist[2*dest];
		p[1].y = input->pointlist[2*dest+1];
		p[2].x = input->pointlist[2*apex];
		p[2].y = input->pointlist[2*apex+1];

		for (int i = 0; i < 3; i++) 
		{
			int j = (i + 1) % 3;
			int k = (i + 2) % 3;
			dx[i] = p[j].x - p[k].x;
			dy[i] = p[j].y - p[k].y;
			edgelength[i] = dx[i] * dx[i] + dy[i] * dy[i];			 
		}

		for (int i = 0; i < 3; i++) 
		{
			int  j = (i + 1) % 3;
			int  k = (i + 2) % 3;
			REAL dotproduct = dx[j] * dx[k] + dy[j] * dy[k];
			REAL cossquare = dotproduct * dotproduct / (edgelength[j] * edgelength[k]);
			if(cossquare > goodAngle)
			{
				printf("Bad triangle %i, smallest angles's cossquare = %f, goodAngle = %f\n",
				num, cossquare, goodAngle);
				return false;
			}
		}
	}

	return true;
}

bool checkResult(triangulateio *input, triangulateio *output,double theta)
{
	printf("Checking result......\n");

	// check input points
	printf("Checking input points......\n");
	for(int i=0; i<input->numberofpoints; i++)
	{
		if(input->pointlist[2*i] != output->pointlist[2*i] ||
			input->pointlist[2*i+1] != output->pointlist[2*i+1])
		{
			printf("Missing input point %d !\n",i);
			return false;
		}
	}

	// check vertices' indices
	printf("Checking indices......\n");
	for(int i=0; i<input->numberoftriangles; i++)
	{
		int index0,index1,index2;
		index0 = input->trianglelist[3*i];
		index1 = input->trianglelist[3*i+1];
		index2 = input->trianglelist[3*i+2];
		if( index0 < 0 || index0 >= input->numberofpoints ||
			index1 < 0 || index1 >= input->numberofpoints ||
			index2 < 0 || index2 >= input->numberofpoints )
		{
			printf("Invalid triangle indices %d: %d, %d, %d\n",
				i, index0,index1,index2);
			return false;
		}
	}

	// check neighbors
	printf("Checking neighbors......\n");
	if(!checkNeighbors(output))
		return false;

	// check quality
	printf("Checking quality......\n");
	if(!checkQuality(output,theta))
		return false;

	// check incircle property
	//printf("Checking incircle property......\n");
	//if(!checkIncircle(output))
	//	return false;

	return true;
}


//************************************
// CPU Functions used by CPU Triangle Method

// CPU main routine
// CPU_Triangle_Quality
// Use Software Triangle to compute quality mesh
void CPU_Triangle_Quality(triangulateio *input, triangulateio *result, double theta, int mode)
{   
	std::ostringstream strs;
	strs << theta;
	std::string angle = strs.str();

	std::string com_str;
	if(mode) // ruppert
		com_str = "pzDQnrq" + angle;
	else // chew
		com_str = "pzQnrq" + angle;
	char *com = new char[com_str.length() + 1];
	strcpy(com, com_str.c_str());

	triangulateio triResult;
    memset(&triResult, 0, sizeof(triangulateio));
	triangulate(com, input, &triResult, NULL); //DT mesh
	memcpy(result, &triResult, sizeof(triangulateio));
}

//************************************
// CPU Functions used to generate input mesh

void saveInput(triangulateio *input, int distribution, int seed, int numberofpoints, int numberofsegs, double min_input_angle)
{
	double angle = min_input_angle;
	if(angle > 60.0)
		angle = 60.0;

	std::ostringstream strs;
	if(angle == 60.0)
		strs << "input/d" << distribution << "_s" << seed << "_p" << numberofpoints << "_c" << numberofsegs;
	else
		strs << "input/d" << distribution << "_s" << seed << "_p" << numberofpoints << "_c" << numberofsegs << "_with_minimum_input_angle_" << angle;
	std::string fn = strs.str();
	char *com = new char[fn.length() + 1];
	strcpy(com, fn.c_str());

	saveCDT(input,com);
}

bool readInput(triangulateio *input, int distribution, int seed, int numberofpoints, int numberofsegs, double min_input_angle)
{

	double angle = min_input_angle;
	if(angle > 60.0)
		angle = 60.0;

	printf("Try to read from file (Minimum input angle = %lf): ", min_input_angle);

	std::ostringstream strs;
	if(angle == 60.0)
		strs << "input/d" << distribution << "_s" << seed << "_p" << numberofpoints << "_c" << numberofsegs;
	else
		strs << "input/d" << distribution << "_s" << seed << "_p" << numberofpoints << "_c" << numberofsegs << "_with_minimum_input_angle_"  << angle;
	std::string fn = strs.str();
	char *com = new char[fn.length() + 1];
	strcpy(com, fn.c_str());

	triangulateio triInput, triResult;
	memset(&triInput, 0, sizeof(triangulateio));
    memset(&triResult, 0, sizeof(triangulateio));
	bool r = readCDT(&triInput,com);
	if(r)
	{
		triangulate("pzQn",&triInput,&triResult, NULL); //DT mesh
		memcpy(input, &triResult, sizeof(triangulateio));
		return true;
	}
	else
		return false;
}

void saveOutput(char * filename, triangulateio *output, int distribution, int seed, int numberofpoints, int numberofsegs, double theta, double runtime)
{
	FILE *fp;
	fp = fopen(filename, "w");

	fprintf(fp, "Number of points = %d\n", output->numberofpoints);
	fprintf(fp, "Number of triangles = %d\n", output->numberoftriangles);
	fprintf(fp, "Number of segments = %d\n", output->numberofsegments);
	fprintf(fp, "Runtime = %lf\n", runtime);
	fclose(fp);
}

bool readOutput(char * filename, int * numberofpoints, int * numberoftriangles, int * numberofsegments, double * runtime)
{
	FILE *fp;
	fp = fopen(filename, "r");

	if(fp == NULL)
	{
		printf("Cannot find the output file\n");
		return false;
	}

	int np,nt,nc;
	np = nt = nc = 0;
	double mytime;
	int ln = 0;
	char buf[100];
	while (fgets(buf, 100, fp) != NULL) {
		int n;
		if(ln == 0)
			n = sscanf(buf, "Number of points = %d", &np);
		else if (ln == 1)
			n = sscanf(buf, "Number of triangles = %d", &nt);
		else if (ln == 2)
			n = sscanf(buf, "Number of segments = %d", &nc);
		else if (ln == 3)
			n = sscanf(buf, "Runtime = %lf", &mytime);
		if(!n)
			break;
		ln++;
	}

	if(numberofpoints != NULL)
		*numberofpoints = np;
	if(numberoftriangles != NULL)
		*numberoftriangles = nt;
	if(numberofsegments != NULL)
		*numberofsegments = nc;
	if(runtime != NULL)
		*runtime = mytime;

	fclose(fp);
	return true;
}

void experiment_statistic()
{
	FILE *fp;
	fp = fopen("result/auto", "w");
	int numOfPoints,distribution,seed,numOfSegments;
	seed = 0;
	for(distribution = 0; distribution <= 3; distribution++)
	{
		for(numOfPoints = 50000; numOfPoints <= 100000; numOfPoints += 10000)
		{
			for(numOfSegments = numOfPoints*0.1; numOfSegments <= numOfPoints*0.5 ; numOfSegments += numOfPoints*0.1)
			{
					if( (distribution == 0 && numOfPoints == 60000) ||
						(distribution == 0 && numOfPoints == 100000 && (numOfSegments == 40000 || numOfSegments == 50000)) ||
						(distribution == 2 && numOfPoints == 60000) ||
						(distribution == 3 && numOfPoints == 70000))
						seed = 1;
					else if ( (distribution == 1 && numOfPoints == 60000) ||
							  (distribution == 2 && numOfPoints == 80000))
						seed = 2;
					else
						seed = 0;
					double theta = 15;
					printf("Numberofpoints = %d, Numberofsegment = %d, Distribution = %d\n",
						numOfPoints, numOfSegments, distribution);

					int r_p[2],r_t[2],r_c[2];
					double runtime[2];

					std::ostringstream strs0;
					strs0 << "result/d" << distribution << "_s" << seed << "_p" << numOfPoints << "_c" << numOfSegments << "_t" << theta << "_cpu";
					std::string fn0 = strs0.str();
					char *com0 = new char[fn0.length() + 1];
					strcpy(com0, fn0.c_str());

					if(!readOutput(com0,&r_p[0], &r_t[0], &r_c[0], &runtime[0]))
						continue;

					std::ostringstream strs1;
					strs1 << "result/d" << distribution << "_s" << seed << "_p" << numOfPoints << "_c" << numOfSegments << "_t" << theta << "_gpu";
					std::string fn1 = strs1.str();
					char *com1 = new char[fn1.length() + 1];
					strcpy(com1, fn1.c_str());
				
					if(!readOutput(com1,&r_p[1], &r_t[1], &r_c[1], &runtime[1]))
						continue;

					//printf("%d %d %d %d %d %d %d %d %d %lf %d %d %d %lf\n",
					//	distribution,seed,numOfPoints,numOfSegments,25,0,
					//	r_p[0],r_t[0],r_c[0],runtime[0],
					//	r_p[1],r_t[1],r_c[1],runtime[1]);

					fprintf(fp,"%d,%d,%d,%d,%d,%d,%d,%d,%d,%lf,%d,%d,%d,%lf\n",
						distribution,seed,numOfPoints,numOfSegments,25,0,
						r_p[0],r_t[0],r_c[0],runtime[0],
						r_p[1],r_t[1],r_c[1],runtime[1]);

			}
			fprintf(fp, "\n");
		}
	}
	fclose(fp);
}

void experiment_statistic_ruppert()
{
	FILE *fp;
	fp = fopen("result/auto_ruppert", "w");
	int numOfPoints,distribution,seed,numOfSegments;
	seed = 0;
	for(distribution = 0; distribution <= 3; distribution++)
	{
		for(numOfPoints = 50000; numOfPoints <= 100000; numOfPoints += 10000)
		{
			for(numOfSegments = numOfPoints*0.1; numOfSegments <= numOfPoints*0.5 ; numOfSegments += numOfPoints*0.1)
			{
				if( (distribution == 0 && numOfPoints == 100000 && (numOfSegments == 40000 || numOfSegments == 50000)) ||
					(distribution == 1 && numOfPoints == 70000 && (numOfSegments == 28000 || numOfSegments == 35000)) ||
					(distribution == 2 && numOfPoints == 80000 && (numOfSegments == 32000 || numOfSegments == 40000)) ||
					(distribution == 3 && numOfPoints == 70000))
					seed = 1;
				else
					seed = 0;

				double theta = 15;
				printf("Numberofpoints = %d, Numberofsegment = %d, Distribution = %d\n",
					numOfPoints, numOfSegments, distribution);

				int r_p[2],r_t[2],r_c[2];
				r_p[0] = r_p[1] = r_t[0] = r_t[1] = r_c[0] = r_c[1] = 0;
				double runtime[2];
				runtime[0] = runtime[1] = 0;

				std::ostringstream strs0;
				strs0 << "result/d" << distribution << "_s" << seed << "_p" << numOfPoints << "_c" << numOfSegments << "_t" << theta << "_cpu_ruppert";
				std::string fn0 = strs0.str();
				char *com0 = new char[fn0.length() + 1];
				strcpy(com0, fn0.c_str());

				if(!readOutput(com0,&r_p[0], &r_t[0], &r_c[0], &runtime[0]))
				{
				}

				std::ostringstream strs1;
				strs1 << "result/d" << distribution << "_s" << seed << "_p" << numOfPoints << "_c" << numOfSegments << "_t" << theta << "_gpu_ruppert";
				std::string fn1 = strs1.str();
				char *com1 = new char[fn1.length() + 1];
				strcpy(com1, fn1.c_str());
				
				if(!readOutput(com1,&r_p[1], &r_t[1], &r_c[1], &runtime[1]))
				{
				}

				//printf("%d %d %d %d %d %d %d %d %d %lf %d %d %d %lf\n",
				//	distribution,seed,numOfPoints,numOfSegments,25,0,
				//	r_p[0],r_t[0],r_c[0],runtime[0],
				//	r_p[1],r_t[1],r_c[1],runtime[1]);

				fprintf(fp,"%d,%d,%d,%d,%d,%d,%d,%d,%d,%lf,%d,%d,%d,%lf\n",
					distribution,seed,numOfPoints,numOfSegments,20,0,
					r_p[0],r_t[0],r_c[0],runtime[0],
					r_p[1],r_t[1],r_c[1],runtime[1]);

			}
			fprintf(fp, "\n");
		}
	}
	fclose(fp);
}

void experiment_input(int mode, double min_input_angle)
{
	int numOfPoints,distribution,seed,numOfSegments;
	for(distribution = 0; distribution <= 3; distribution++)
	{
		for(numOfPoints = 50000; numOfPoints <= 100000; numOfPoints += 10000)
		{
			for(numOfSegments = numOfPoints*0.1; numOfSegments <= numOfPoints*0.5 ; numOfSegments += numOfPoints*0.1)
			{
				if(mode)
				{
					if( (distribution == 0 && numOfPoints == 90000)  ||
						(distribution == 1 && numOfPoints == 60000)  ||
						(distribution == 2 && numOfPoints == 80000)  ||
						(distribution == 3 && numOfPoints == 70000))
						seed = 1;
					else
						seed = 0;
				}
				else
				{
					if( (distribution == 0 && (numOfPoints == 60000 || numOfPoints == 90000)) ||
						(distribution == 2 && numOfPoints == 60000) ||
						(distribution == 3 && numOfPoints == 70000))
						seed = 1;
					else if ( (distribution == 1 && numOfPoints == 60000) ||
								(distribution == 2 && numOfPoints == 80000))
						seed = 2;
					else
						seed = 0;
				}

				printf("Numberofpoints = %d, Numberofsegment = %d, Distribution = %d, Seed = %d\n",
					numOfPoints, numOfSegments, distribution, seed);

				std::ostringstream strs;
				if(min_input_angle == 60.0)
					strs << "input/d" << distribution << "_s" << seed << "_p" << numOfPoints << "_c" << numOfSegments;
				else
					strs << "input/d" << distribution << "_s" << seed << "_p" << numOfPoints << "_c" << numOfSegments << "_with_minimum_input_angle_"  << min_input_angle;
				std::string fn = strs.str();
				char *com = new char[fn.length() + 1];
				strcpy(com, fn.c_str());

				FILE *fp;
				fp = fopen(com, "r");

				triangulateio triInput;
				if(fp == NULL)
				{
					printf("Failed to find input file, start generating...\n");
					GenerateRandomInput(numOfPoints,numOfSegments,seed,distribution,&triInput,min_input_angle);
					saveInput(&triInput,distribution,seed,numOfPoints,numOfSegments,min_input_angle);
					delete[] triInput.pointlist;
					delete[] triInput.trianglelist;
					delete[] triInput.segmentlist;
				}
				else
				{
					printf("Found input file, Skip!\n");
					fclose(fp);
				}
						
				printf("\n");
			}
		}
	}
}

void experiment_triangle(int mode, double min_allowable_angle, double min_input_angle)
{
	int numOfPoints,distribution,seed,numOfSegments;
	seed = 0;
	for(distribution = 0; distribution <= 3; distribution++)
	{
		for(numOfPoints = 50000; numOfPoints <= 100000; numOfPoints += 10000)
		{
			for(numOfSegments = numOfPoints*0.1; numOfSegments <= numOfPoints*0.5 ; numOfSegments += numOfPoints*0.1)
			{
				if(true)
				{
					//if( (distribution == 0 && numOfPoints == 60000) ||
					//	(distribution == 0 && numOfPoints == 100000 && (numOfSegments == 40000 || numOfSegments == 50000)) ||
					//	(distribution == 2 && numOfPoints == 60000) ||
					//	(distribution == 3 && numOfPoints == 70000))
					//	seed = 1;
					//else if ( (distribution == 1 && numOfPoints == 60000) ||
					//		  (distribution == 2 && numOfPoints == 80000))
					//	seed = 2;
					//else
					//	seed = 0;

					printf("Random Input: Numberofpoints = %d, Numberofsegment = %d, Distribution = %d, Seed = %d\n",
						numOfPoints, numOfSegments, distribution, seed);

					printf("Running Mode: %s, minimum allowable angle = %lf, minimum input angle = %lf\n",mode? "Ruppert":"Chew",
						min_allowable_angle, min_input_angle);

					triangulateio triInput;
					std::ostringstream strs;
					strs << "result/d" << distribution << "_s" << seed << "_p" << numOfPoints << "_c" << numOfSegments << "_t" << min_allowable_angle << "_cpu";
					if(mode)
						strs << "Ruppert";
					if(min_input_angle < 60.0)
						strs << "with_minimum_input_angle_" << min_input_angle;
					std::string fn = strs.str();
					char *com = new char[fn.length() + 1];
					strcpy(com, fn.c_str());
					if(readOutput(com,NULL, NULL, NULL, NULL))
					{
						printf("Find the output file, Skip!\n");
					}
					else if(!readInput(&triInput,distribution,seed,numOfPoints,numOfSegments,min_input_angle))
					{
						printf("Don't find the input file, Skip!\n");
					}
					else
					{
						printf("Triangle is running...\n");
						StopWatchInterface *timer = 0; // timer
						sdkCreateTimer( &timer );
						double cpu_time;
						sdkResetTimer( &timer );
						sdkStartTimer( &timer );
						triangulateio cpu_result;
						CPU_Triangle_Quality(&triInput,&cpu_result,min_allowable_angle,mode);
						sdkStopTimer( &timer );
						cpu_time = sdkGetTimerValue( &timer );
						saveOutput(com,&cpu_result,distribution,seed,numOfPoints,numOfSegments,min_allowable_angle,cpu_time);
						delete[] triInput.pointlist;
						delete[] triInput.trianglelist;
						delete[] triInput.neighborlist;
						delete[] triInput.segmentlist;
						delete[] cpu_result.pointlist;
						delete[] cpu_result.trianglelist;
						delete[] cpu_result.neighborlist;
						delete[] cpu_result.segmentlist;
					}
					
					printf("\n");
				}
			}
		}
	}
}

void experiment_triangle_ruppert(double angle)
{
	double min_input_angle = 15.0;
	int numOfPoints,distribution,seed,numOfSegments;
	seed = 0;
	for(distribution = 0; distribution <= 3; distribution++)
	{
		for(numOfPoints = 50000; numOfPoints <= 100000; numOfPoints += 10000)
		{
			for(numOfSegments = numOfPoints*0.1; numOfSegments <= numOfPoints*0.5 ; numOfSegments += numOfPoints*0.1)
			{
				if( true )
				{
					if( //(distribution == 0 && numOfPoints == 100000 && (numOfSegments == 40000 || numOfSegments == 50000)) ||
						(distribution == 1 && numOfPoints == 70000 && (numOfSegments == 28000 || numOfSegments == 35000)) ||
						(distribution == 2 && numOfPoints == 80000 && (numOfSegments == 32000 || numOfSegments == 40000)) ||
						(distribution == 3 && numOfPoints == 70000))
						seed = 1;
					else if ((distribution == 3 && numOfPoints == 90000 && numOfSegments == 45000))
						seed = (angle == 20) ? 0:2;
					else
						seed = 0;
					double theta = angle;

					printf("Numberofpoints = %d, Numberofsegment = %d, Distribution = %d\n",
						numOfPoints, numOfSegments, distribution);
					triangulateio triInput;
					std::ostringstream strs;
					strs << "result/d" << distribution << "_s" << seed << "_p" << numOfPoints << "_c" << numOfSegments << "_t" << theta << "_cpu_ruppert";
					std::string fn = strs.str();
					char *com = new char[fn.length() + 1];
					strcpy(com, fn.c_str());
					if(readOutput(com, NULL, NULL, NULL, NULL))
					{
						printf("Find the output file, Skip!\n");
					}
					else if(!readInput(&triInput,distribution,seed,numOfPoints,numOfSegments,min_input_angle))
					{
						printf("Don't find the input file, Skip!\n");
					}
					else
					{
						printf("Triangle is running...\n");
						StopWatchInterface *timer = 0; // timer
						sdkCreateTimer( &timer );
						double cpu_time;
						sdkResetTimer( &timer );
						sdkStartTimer( &timer );
						triangulateio cpu_result;
						CPU_Triangle_Quality(&triInput,&cpu_result,theta,1);
						sdkStopTimer( &timer );
						cpu_time = sdkGetTimerValue( &timer );
						saveOutput(com,&cpu_result,distribution,seed,numOfPoints,numOfSegments,theta,cpu_time);
						delete[] triInput.pointlist;
						delete[] triInput.trianglelist;
						delete[] triInput.neighborlist;
						delete[] triInput.segmentlist;
						delete[] cpu_result.pointlist;
						delete[] cpu_result.trianglelist;
						delete[] cpu_result.neighborlist;
						delete[] cpu_result.segmentlist;
					}
					
					printf("\n");
				}
			}
		}
	}
}

void experiment_gQM()
{
	double min_input_angle = 15.0;
	int numOfPoints,distribution,seed,numOfSegments;
	seed = 0;
	bool firstTime = true;
	for(distribution = 0; distribution <= 3; distribution++)
	{
		for(numOfPoints = 50000; numOfPoints <= 100000; numOfPoints += 10000)
		{
			for(numOfSegments = numOfPoints*0.1; numOfSegments <= numOfPoints*0.5 ; numOfSegments += numOfPoints*0.1)
			{
				if(true)
				{
					if( (distribution == 0 && numOfPoints == 60000) ||
						(distribution == 0 && numOfPoints == 100000 && (numOfSegments == 40000 || numOfSegments == 50000)) ||
						(distribution == 2 && numOfPoints == 60000) ||
						(distribution == 3 && numOfPoints == 70000))
						seed = 1;
					else if ( (distribution == 1 && numOfPoints == 60000) ||
							  (distribution == 2 && numOfPoints == 80000))
						seed = 2;
					else
						seed = 0;
					double theta = 15;
					printf("Numberofpoints = %d, Numberofsegment = %d, Distribution = %d, Seed = %d\n",
						numOfPoints, numOfSegments, distribution, seed);
					triangulateio triInput;
					std::ostringstream strs;
					strs << "result/d" << distribution << "_s" << seed << "_p" << numOfPoints << "_c" << numOfSegments << "_t" << theta << "_gpu";
					std::string fn = strs.str();
					char *com = new char[fn.length() + 1];
					strcpy(com, fn.c_str());
					if(readOutput(com,NULL, NULL, NULL, NULL))
					{
						printf("Find the output file, Skip!\n");
					}
					else if(!readInput(&triInput,distribution,seed,numOfPoints,numOfSegments,min_input_angle))
					{
						printf("Don't find the input file, Skip!\n");
					}
					else
					{
						printf("gQM is running...\n");
						//Sleep(1000);
						StopWatchInterface *timer = 0; // timer
						sdkCreateTimer( &timer );
						double gpu_time;
						//printf("%d, %d, %d\n",triInput.numberofpoints,triInput.numberofsegments,triInput.numberoftriangles);
						sdkResetTimer( &timer );
						sdkStartTimer( &timer );
						triangulateio gpu_result;
						InsertPolicy insertpolicy = Offcenter;
						DeletePolicy deletepolicy = Connected;
						GPU_Refine_Quality(&triInput,&gpu_result,theta, insertpolicy,deletepolicy,0,-1,
							NULL,NULL);
						sdkStopTimer( &timer );
						gpu_time = sdkGetTimerValue( &timer );
						saveOutput(com,&gpu_result,distribution,seed,numOfPoints,numOfSegments,theta,gpu_time);
						delete[] triInput.pointlist;
						delete[] triInput.trianglelist;
						delete[] triInput.neighborlist;
						delete[] triInput.segmentlist;
						delete[] gpu_result.pointlist;
						delete[] gpu_result.trianglelist;
						delete[] gpu_result.neighborlist;
						delete[] gpu_result.segmentlist;
					}
					
					printf("\n");
				}
			}
		}
	}
}

void experiment_gQM_ruppert()
{
	double min_input_angle = 5.0;
	int numOfPoints,distribution,seed,numOfSegments;
	seed = 0;
	bool firstTime = true;
	for(distribution = 0; distribution <= 3; distribution++)
	{
		for(numOfPoints = 50000; numOfPoints <= 100000; numOfPoints += 10000)
		{
			for(numOfSegments = numOfPoints*0.1; numOfSegments <= numOfPoints*0.5 ; numOfSegments += numOfPoints*0.1)
			{
				if( true )
				{
					if( //(distribution == 0 && numOfPoints == 100000 && numOfSegments == 40000) ||
						(distribution == 1 && numOfPoints == 70000 && (numOfSegments == 28000 || numOfSegments == 35000)) ||
						(distribution == 2 && numOfPoints == 80000 && (numOfSegments == 32000 || numOfSegments == 40000)) ||
						(distribution == 3 && numOfPoints == 70000))
						seed = 1;
					else if ((distribution == 3 && numOfPoints == 90000 && numOfSegments == 45000))
						seed = 2;
					else if (distribution == 3 && numOfPoints == 100000 && numOfSegments == 50000)
						seed = 3;
					else
						seed = 0;
					double theta = 15;
					printf("Numberofpoints = %d, Numberofsegment = %d, Distribution = %d, Seed = %d\n",
						numOfPoints, numOfSegments, distribution, seed);
					triangulateio triInput;
					std::ostringstream strs;
					strs << "result/d" << distribution << "_s" << seed << "_p" << numOfPoints << "_c" << numOfSegments << "_t" << theta << "_gpu_ruppert";
					std::string fn = strs.str();
					char *com = new char[fn.length() + 1];
					strcpy(com, fn.c_str());
					if(readOutput(com,NULL, NULL, NULL, NULL))
					{
						printf("Find the output file, Skip!\n");
					}
					else if(!readInput(&triInput,distribution,seed,numOfPoints,numOfSegments,min_input_angle))
					{
						printf("Don't find the input file, Skip!\n");
					}
					else
					{
						printf("gQM is running...\n");
						//Sleep(1000);
						StopWatchInterface *timer = 0; // timer
						sdkCreateTimer( &timer );
						double gpu_time;
						//printf("%d, %d, %d\n",triInput.numberofpoints,triInput.numberofsegments,triInput.numberoftriangles);
						sdkResetTimer( &timer );
						sdkStartTimer( &timer );
						triangulateio gpu_result;
						InsertPolicy insertpolicy = Offcenter;
						DeletePolicy deletepolicy = Connected;
						GPU_Refine_Quality(&triInput,&gpu_result,theta, insertpolicy,deletepolicy,1,-1,
							NULL,NULL);
						sdkStopTimer( &timer );
						gpu_time = sdkGetTimerValue( &timer );
						saveOutput(com,&gpu_result,distribution,seed,numOfPoints,numOfSegments,theta,gpu_time);
						delete[] triInput.pointlist;
						delete[] triInput.trianglelist;
						delete[] triInput.neighborlist;
						delete[] triInput.segmentlist;
						delete[] gpu_result.pointlist;
						delete[] gpu_result.trianglelist;
						delete[] gpu_result.neighborlist;
						delete[] gpu_result.segmentlist;
					}
					
					printf("\n");
				}
			}
		}
	}
}

void generateTikzpicture(char * filename, int numofinputpoints, triangulateio * output)
{
	FILE *fp;
	fp = fopen(filename, "w");

	bool inputPoints = false;
	bool steinerPoints = false;
	bool segments = true;
	bool edges = true;
	
	// draw input points with black color
	if(inputPoints)
	{
		fprintf(fp, "\\filldraw [black]\n");
		for( int i =0; i < numofinputpoints; i++)
		{
			if(i != numofinputpoints - 1)
				fprintf(fp, "(%f,%f) circle [radius=3pt]\n",output->pointlist[2*i],output->pointlist[2*i+1]);
			else
				fprintf(fp, "(%f,%f) circle [radius=3pt];\n",output->pointlist[2*i],output->pointlist[2*i+1]);
		}
	}

	// draw Steiner points with red color
	if(steinerPoints)
	{
		fprintf(fp, "\\filldraw [red]\n");
		for( int i = numofinputpoints; i < output->numberofpoints; i++)
		{
			if(i != output->numberofpoints - 1)
				fprintf(fp, "(%f,%f) circle [radius=3pt]\n",output->pointlist[2*i],output->pointlist[2*i+1]);
			else
				fprintf(fp, "(%f,%f) circle [radius=3pt];\n",output->pointlist[2*i],output->pointlist[2*i+1]);
		}
	}

	// draw all edges with black color
	if(edges)
	{
		fprintf(fp, "\\draw [line width=0mm, black]\n");
		for( int i = 0; i < output->numberoftriangles; i++)
		{
			int index[3] = 
			{
				output->trianglelist[3*i],
				output->trianglelist[3*i+1],
				output->trianglelist[3*i+2]
			};
			fprintf(fp, "(%f,%f) -- (%f,%f)\n",
				output->pointlist[2*index[0]],output->pointlist[2*index[0]+1],
				output->pointlist[2*index[1]],output->pointlist[2*index[1]+1]);
			fprintf(fp, "(%f,%f) -- (%f,%f)\n",
				output->pointlist[2*index[0]],output->pointlist[2*index[0]+1],
				output->pointlist[2*index[2]],output->pointlist[2*index[2]+1]);
			if(i != output->numberoftriangles -1)
				fprintf(fp, "(%f,%f) -- (%f,%f)\n",
					output->pointlist[2*index[1]],output->pointlist[2*index[1]+1],
					output->pointlist[2*index[2]],output->pointlist[2*index[2]+1]);
			else
				fprintf(fp, "(%f,%f) -- (%f,%f);\n",
					output->pointlist[2*index[1]],output->pointlist[2*index[1]+1],
					output->pointlist[2*index[2]],output->pointlist[2*index[2]+1]);
		}
	}

	// draw segments with blue colr
	if(segments)
	{
		fprintf(fp, "\\draw [red,thick]\n");
		for( int i = 0; i < output->numberofsegments; i++)
		{
			int index[2] = 
			{
				output->segmentlist[2*i],
				output->segmentlist[2*i+1]
			};
			if(i != output->numberofsegments -1)
				fprintf(fp, "(%f,%f) -- (%f,%f)\n",
					output->pointlist[2*index[0]],output->pointlist[2*index[0]+1],
					output->pointlist[2*index[1]],output->pointlist[2*index[1]+1]);
			else
				fprintf(fp, "(%f,%f) -- (%f,%f);\n",
					output->pointlist[2*index[0]],output->pointlist[2*index[0]+1],
					output->pointlist[2*index[1]],output->pointlist[2*index[1]+1]);
		}
	}

	fclose(fp);
}

void checkInputAngles(triangulateio * input)
{
	REAL goodAngle = cos(60.0 * PI/180.0);
	goodAngle *= goodAngle;
	for(int j = 0; j < input->numberofsegments; j++)
	{
		int p1,p2;
		p1 = input->segmentlist[2*j];
		p2 = input->segmentlist[2*j+1];
		bool pass = true;
		for(int i=0; i < input->numberofsegments; i++)
		{
			if( i == j)
				continue;

			int tmp1,tmp2;
			tmp1 = input->segmentlist[2*i];
			tmp2 = input->segmentlist[2*i+1];

			if(tmp1 == p1 || tmp1 == p2 || tmp2 == p1 || tmp2 == p2)
			{
				REAL dx[2], dy[2], edgelength[2];
				REAL2 v[] =
				{
					MAKE_REAL2(input->pointlist[2*p1],input->pointlist[2*p1+1]),
					MAKE_REAL2(input->pointlist[2*p2],input->pointlist[2*p2+1]),
					MAKE_REAL2(input->pointlist[2*tmp1],input->pointlist[2*tmp1+1]),
					MAKE_REAL2(input->pointlist[2*tmp2],input->pointlist[2*tmp2+1])
				};

				edgelength[0] = (v[0].x-v[1].x)*(v[0].x-v[1].x) + 
					(v[0].y - v[1].y)*(v[0].y-v[1].y);
				edgelength[1] = (v[2].x-v[3].x)*(v[2].x-v[3].x) + 
					(v[2].y - v[3].y)*(v[2].y-v[3].y);
				if( p1 == tmp1 )
				{
					dx[0] = v[1].x - v[0].x;
					dx[1] = v[3].x - v[2].x;
					dy[0] = v[1].y - v[0].y;
					dy[1] = v[3].y - v[2].y;
				}
				else if ( p1 == tmp2)
				{
					dx[0] = v[1].x - v[0].x;
					dx[1] = v[2].x - v[3].x;
					dy[0] = v[1].y - v[0].y;
					dy[1] = v[2].y - v[3].y;
				}
				else if ( p2 == tmp1)
				{
					dx[0] = v[0].x - v[1].x;
					dx[1] = v[3].x - v[2].x;
					dy[0] = v[0].y - v[1].y;
					dy[1] = v[3].y - v[2].y;
				}
				else
				{
					dx[0] = v[0].x - v[1].x;
					dx[1] = v[2].x - v[3].x;
					dy[0] = v[0].y - v[1].y;
					dy[1] = v[2].y - v[3].y;
				}
				REAL dotproduct = dx[0]*dx[1]+dy[0]*dy[1];
				REAL cossquare = dotproduct*dotproduct /(edgelength[0]*edgelength[1]);
				if( dotproduct > 0 && cossquare > goodAngle)
				{
					// the angle between two segment is smaller than 60 degree
					printf("Segment %d and %d: small angle formed.\n", i,j);
					return;
				}
			}
		}
	}
}

void readRealData_Binary(char * file_vertex, char * file_constraint, int offset_seg, int num_seg, int extra_seg, triangulateio * result)
{
	triangulateio triInput;
	memset(&triInput, 0, sizeof(triangulateio));

	printf("Try to read from file: ");

	// Read points first
	FILE *fp;
	fopen_s(&fp,file_vertex, "rb");

	if(fp == NULL)
	{
		printf("Cannot find the input file\n");
		return;
	}
	else
		printf("Succeed\n");

	int numofpoints;
	fread(&numofpoints, sizeof(int), 1, fp);

	float * pointlist = new float[2*numofpoints];
	fread(pointlist,sizeof(float),2*numofpoints,fp);

	fclose(fp);

	// Read constraints
	fopen_s(&fp,file_constraint, "rb");

	if(fp == NULL)
	{
		printf("Cannot find the input file\n");
		return;
	}

	int numofsegs = 0; 
	fread(&numofsegs, sizeof(int), 1, fp);

	int * segmentlist = new int[numofsegs*2];

	fread(segmentlist,sizeof(int),numofsegs*2,fp);

	fclose(fp);

	// Extract and set up wanted input

	int * marker = new int[numofpoints];

	for(int i=0; i<numofpoints; i++)
		marker[i] = 0;

	if(offset_seg >= numofsegs)
	{
		printf("Offset exceeds the number of segments\n");
		exit(0);
	}
	else if ( offset_seg + num_seg + extra_seg > numofsegs )
	{
		printf("Last element exceeds the number of segments, resize the element size\n");
		exit(0);
	}

	for(int i=0; i<num_seg + extra_seg; i++)
	{
		int segid = offset_seg + i;
		int p1 = segmentlist[2*segid];
		int p2 = segmentlist[2*segid+1];
		marker[p1] = 1;
		marker[p2] = 1;
	}

	int * scan = new int[numofpoints];

	for(int i=0; i<numofpoints; i++)
	{
		// inclusive scan
		if(i == 0)
			scan[i] = marker[i];
		else
			scan[i] = scan[i-1] + marker[i];
	}

	int num_point = scan[numofpoints-1]; // the number of marked points

	triInput.numberofpoints = num_point;
	triInput.pointlist = new double[2*num_point];
	for(int i=0; i<numofpoints; i++)
	{
		if(marker[i] == 1)
		{
			int newIndex = scan[i] - 1;
			triInput.pointlist[2*newIndex] = pointlist[2*i];
			triInput.pointlist[2*newIndex+1] = pointlist[2*i+1];
		}
	}

	//triInput.numberofpoints = numofpoints;
	//triInput.pointlist = new double[2*numofpoints];
	//for(int i=0; i<numofpoints; i++)
	//{
	//	triInput.pointlist[2*i] = pointlist[2*i];
	//	triInput.pointlist[2*i+1] = pointlist[2*i+1];
	//}

	triInput.numberofsegments = num_seg;
	triInput.segmentlist = new int[2*num_seg];
	for(int i=0; i<num_seg; i++)
	{
		int segid = offset_seg + i;
		int p1 = segmentlist[2*segid];
		int p2 = segmentlist[2*segid+1];
		int new_p1 = scan[p1] - 1;
		int new_p2 = scan[p2] - 1;
		triInput.segmentlist[2*i] = new_p1;
		triInput.segmentlist[2*i+1] = new_p2;
	}

	//triInput.numberofsegments = num_seg;
	//triInput.segmentlist = new int[2*num_seg];
	//for(int i=0; i<num_seg; i++)
	//{
	//	int segid = offset_seg + i;
	//	int p1 = segmentlist[2*segid];
	//	int p2 = segmentlist[2*segid+1];
	//	triInput.segmentlist[2*i] = p1;
	//	triInput.segmentlist[2*i+1] = p2;
	//}

	// Compute CDT
	triangulateio triCDT;
	memset(&triCDT, 0, sizeof(triangulateio));
	triangulate("pzQnc", &triInput, &triCDT, NULL);

	memcpy(result, &triCDT, sizeof(triangulateio));
}

void processRealData(triangulateio * result)
{
	triangulateio triInput;
	memset(&triInput, 0, sizeof(triangulateio));

	// Read points first
	printf("Reading points...\n");
	FILE *fp;
	fp = fopen("realworld/lena.tif_vertex.txt", "r");

	if(fp == NULL)
	{
		printf("Cannot find the input file\n");
		return;
	}

	char buf[100];
	
	int ln = 0;

	while (fgets(buf, 100, fp) != NULL) {
		if(ln == 0)
		{
			if(sscanf(buf,"%d",&(triInput.numberofpoints)) != 1)
			{
				printf("Incorrect format\n");
				fclose(fp);
				return;
			}
			else
				triInput.pointlist = new double[2*triInput.numberofpoints];
		}
		else if(ln < triInput.numberofpoints + 1)
		{
			int tmp;
			double x,y;
			if(sscanf(buf,"%d %lf %lf",
				&tmp,&x,&y) != 3)
			{
				printf("Incorrect format\n");
				fclose(fp);
				return;
			}
			*(triInput.pointlist + 2*(ln-1)) = x*1.0;
			*(triInput.pointlist + 2*(ln-1)+1) = y*1.0;
		}
		else
			break;
		ln++;
	}

	fclose(fp);

	// Read constraints
	printf("Reading constraints...\n");
	fp = fopen("realworld/lena.tif_constraint.txt", "r");

	if(fp == NULL)
	{
		printf("Cannot find the input file\n");
		return;
	}

	ln = 0;
	int numofseg_file = 0;
	int * segmentlist_file;

	while (fgets(buf, 100, fp) != NULL) {
		if(ln == 0)
		{
			if(sscanf(buf,"%d",&numofseg_file) != 1)
			{
				printf("Incorrect format\n");
				fclose(fp);
				return;
			}
			else
			{
				segmentlist_file = new int[2*numofseg_file];
				triInput.segmentlist = new int[2*numofseg_file];
				triInput.numberofsegments = numofseg_file;
			}
		}
		else if(ln < numofseg_file + 1)
		{
			int id, p1, p2;
			if(sscanf(buf,"%d %d %d",&id,&p1,&p2) != 3)
			{
				printf("Incorrect format\n");
				fclose(fp);
				return;
			}
			segmentlist_file[2*(ln-1)] = p1;
			segmentlist_file[2*(ln-1)+1] = p2;
			triInput.segmentlist[2*(ln-1)] = p1;
			triInput.segmentlist[2*(ln-1)+1] = p2;
		}
		else
		{
			break;
		}
		ln++;
	}

	fclose(fp);

	//printf("number of segments in file = %d\n", numofseg_file);

	// Compute CDT
	printf("Compute CDT...\n");
	triangulateio triCDT;
	memset(&triCDT, 0, sizeof(triangulateio));
	triangulate("pzQnc", &triInput, &triCDT, NULL);

	//generateTikzpicture("realworld/lena_input",triCDT.numberofpoints,&triCDT);

	// check input point
	for(int i=0; i<triInput.numberofpoints; i++)
	{
		double x1,y1;
		x1 = triInput.pointlist[2*i];
		y1 = triInput.pointlist[2*i+1];
		for(int j=i+1; j<triInput.numberofpoints; j++)
		{
			double x2,y2;
			x2 = triInput.pointlist[2*j];
			y2 = triInput.pointlist[2*j+1];
			if(x1 == x2 && y1 == y2)
				printf("Duplicate input point\n");
		}
	}

	// check input segmenet
	for(int i=0; i<triInput.numberofsegments; i++)
	{
		int p1,p2;
		p1 = triInput.segmentlist[2*i];
		p2 = triInput.segmentlist[2*i+1];
		for(int j=i+1; j<triInput.numberofsegments; j++)
		{
			int e1,e2;
			e1 = triInput.segmentlist[2*j];
			e2 = triInput.segmentlist[2*j+1];
			if( (p1 == e1 && p2 == e2) ||
				(p1 == e2 && p2 == e1))
			{
				printf("Duplicate input endpoints\n");
			}
		}
	}

	// Refinement using Triangle
	printf("Refine in CPU ...\n");
	triangulateio triCPU;
	CPU_Triangle_Quality(&triCDT,&triCPU,15,1);
	//generateTikzpicture("realworld/lena_cpu_chew",triCPU.numberofpoints,&triCPU);

	// Refinement using gQM
	printf("Refine in GPU ...\n");
	triangulateio triGPU;
	InsertPolicy insertpolicy = Offcenter;
	DeletePolicy deletepolicy = Connected;
	GPU_Refine_Quality(&triCDT,&triGPU,15,insertpolicy,deletepolicy,1,-1,
		&debug_ps,&debug_ts);

	// Read from file
	//readCDT(&triGPU,"realworld/lena_GPU");

	// Scaling back
	//saveCDT(&triGPU,"realworld/lena_GPU");
	//for(int i=0; i<triGPU.numberofpoints; i++)
	//{
	//	triGPU.pointlist[2*i] /= 1.5;
	//	triGPU.pointlist[2*i+1] /= 1.5;
	//}

	// Comparsion
	printf("CPU: number of points = %d\n",triCPU.numberofpoints);
	printf("GPU: number of points = %d\n",triGPU.numberofpoints);
	//if(debug_ps != NULL)
	//{
	//	int mid = 0;
	//	for(int i=0; i<triGPU.numberofpoints; i++)
	//	{
	//		if(debug_ps[i].isSegmentSplit())
	//			mid++;
	//	}
	//	printf(" mid = %d\n",mid);
	//}

	memcpy(result, &triGPU, sizeof(triangulateio));

	return;
}

//************************************
// CPU Functions used by OpenGL

void drawTriangles()
{
	if(draw_result->trianglelist != NULL)
	{
		
		for(int i=0; i<draw_result->numberoftriangles;i++)
		{

			
			int index_1,index_2,index_3;
			double x_1,y_1,x_2,y_2,x_3,y_3;

			index_1 = draw_result->trianglelist[3*i];
			index_2 = draw_result->trianglelist[3*i+1];
			index_3 = draw_result->trianglelist[3*i+2];

			x_1 = draw_result->pointlist[2*index_1];
			y_1 = draw_result->pointlist[2*index_1+1];
			x_2 = draw_result->pointlist[2*index_2];
			y_2 = draw_result->pointlist[2*index_2+1];
			x_3 = draw_result->pointlist[2*index_3];
			y_3 = draw_result->pointlist[2*index_3+1];

			triVertex vOrg(x_1,y_1);
			triVertex vDest(x_2,y_2);
			triVertex vApex(x_3,y_3);

			//if(isPureBadTriangle(vOrg, vDest, vApex, 15))
			//{
			//	printf("Pure bad triangle %d: %d(%lf,%lf), %d(%lf,%lf), %d(%lf,%lf)\n",
			//		i,index_1,x_1,y_1,index_2,x_2,y_2,index_3,x_3,y_3);
			//	glColor3f(0.0,1.0,0.0);
			//	glBegin(GL_TRIANGLES);
			//		glVertex2f(x_1,y_1);
			//		glVertex2f(x_2,y_2);
			//		glVertex2f(x_3,y_3);
			//	glEnd();
			//}
			
			glColor3f(0.0,0.0,0.0);
			glBegin(GL_LINES);
				glVertex2f(x_1,y_1);
				glVertex2f(x_2,y_2);
				glVertex2f(x_2,y_2);
				glVertex2f(x_3,y_3);
				glVertex2f(x_3,y_3);
				glVertex2f(x_1,y_1);
			glEnd();
		}
	}
}

void drawSegments()
{
	if(draw_result->segmentlist != NULL)
	{
		glColor3f(0.0,0.0,0.0);
		for(int i=0; i<draw_result->numberofsegments;i++)
		{
			int index_1,index_2;
			double x_1,y_1,x_2,y_2;

			index_1 = draw_result->segmentlist[2*i];
			index_2 = draw_result->segmentlist[2*i+1];

			x_1 = draw_result->pointlist[2*index_1];
			y_1 = draw_result->pointlist[2*index_1+1];
			x_2 = draw_result->pointlist[2*index_2];
			y_2 = draw_result->pointlist[2*index_2+1];

			if(i == -1)
			{
				glColor3f(0.0,0.0,1.0);
				glLineWidth(2);
				glBegin(GL_LINES);
					glVertex2f(x_1,y_1);
					glVertex2f(x_2,y_2);
				glEnd();
			}
			else
			{
				glColor3f(1.0,0.0,0.0);
				glLineWidth(2);
				glBegin(GL_LINES);
					glVertex2f(x_1,y_1);
					glVertex2f(x_2,y_2);
				glEnd();
			}
		}
	}
}

void drawPoints()
{
	if( draw_result->pointlist != NULL)
	{
		glPointSize(3);
		for(int i=0; i<draw_result->numberofpoints;i++)
		{
			double x = draw_result->pointlist[2*i];
			double y = draw_result->pointlist[2*i+1];

			if ( i < draw_numofpoints)
				glColor3f(0.0,0.0,0.0);
			else if ( debug_ps!=NULL && debug_ps[i].isSteiner() && !debug_ps[i].isSegmentSplit()) // for debug
				glColor3f(1.0,0.0,0.0);
			else
				glColor3f(0.0,1.0,0.0);

			glBegin(GL_POINTS);
				glVertex2f(x,y);
			glEnd();
		}
	}
}

void drawSegmentsPoints()
{
	if (draw_result->segmentlist != NULL)
	{
		glPointSize(3);
		for(int i=0; i<draw_result->numberofsegments;i++)
		{
			int index_1,index_2;
			double x_1,y_1,x_2,y_2;

			index_1 = draw_result->segmentlist[2*i];
			index_2 = draw_result->segmentlist[2*i+1];

			x_1 = draw_result->pointlist[2*index_1];
			y_1 = draw_result->pointlist[2*index_1+1];
			x_2 = draw_result->pointlist[2*index_2];
			y_2 = draw_result->pointlist[2*index_2+1];

			glColor3f(1.0,0.0,0.0);
			glBegin(GL_POINTS);
				glVertex2f(x_1,y_1);
				glVertex2f(x_2,y_2);
			glEnd();
		}
	}
}

void init(void) 
{
   glClearColor (1.0, 1.0, 1.0, 0.0);
   glShadeModel (GL_FLAT);
}

void display(void)
{
   glClear (GL_COLOR_BUFFER_BIT);
   //glTranslatef(-5800.0f,-5800.0f,0.0f);
   //glScalef(12.0f,12.0f,0.f);
   drawTriangles();
   drawSegments();
   //drawPoints();
   //drawSegmentsPoints();
   
   glFlush();
}

void reshape (int w, int h)
{
   glViewport (0, 0, (GLsizei) w, (GLsizei) h);
   glMatrixMode (GL_PROJECTION);
   glLoadIdentity ();
   gluOrtho2D (0.0, (GLdouble) w, 0.0, (GLdouble) h);
}

void drawTriangulation(int argc, char** argv, triangulateio * input)
{
	draw_result = input;
	draw_numofpoints = 6117;

	glutInit(&argc, argv);
	glutInitDisplayMode (GLUT_SINGLE | GLUT_RGB);
	glutInitWindowSize (800, 800); 
	glutInitWindowPosition (100, 100);
	glutCreateWindow (argv[0]);
	init ();
	glutDisplayFunc(display);
	glutReshapeFunc(reshape);
	glutMainLoop();
}

/**
 * Host main routine
 */
int
main(int argc, char** argv)
{
	// -1. This paret is for experiment
	//experiment_input(1,5);
	//experiment_input(1,4);
	//experiment_input(1,3);
	//experiment_input(1,2);
	//experiment_input(1,1);

	//experiment_input(0,5);
	//experiment_input(0,4);
	//experiment_input(0,3);
	//experiment_input(0,2);
	//experiment_input(0,1);

	//experiment_statistic();
	//experiment_statistic_ruppert();

	//experiment_triangle(15);
	//experiment_triangle_ruppert(20);
	//experiment_triangle(20);
	//experiment_triangle_ruppert(15);
	

	//experiment_gQM();
	//experiment_gQM_ruppert();

	//return 0;

	// Set up variables
	printf("--------------------------------------------------------\n");
	// constrain
	double theta = 20;

	// running mode
	int mode = 1; // 1: Ruppert, 0: Chew

	printf("Minimum allowable angle = %lf, running mode = %s\n",theta,mode ? "Ruppert":"Chew");

	// running control
	bool real_data = false;
	bool run_cpu = false;
	bool run_gpu = true;
	bool check_result = false;
	int draw = 0; // 0:Dont draw, 1: CPU, 2: GPU

	// policy: this one seems unnecessary for current version
	InsertPolicy insertpolicy = Offcenter;
	DeletePolicy deletepolicy = Connected;

	// 0. Prepare Input
	// Two choices: random input or real date
	printf("--------------------------------------------------------\n");
	printf("0. Preparing for Input Mesh: ");
	triangulateio triInput;
	memset(&triInput, 0, sizeof(triangulateio));

	if(real_data)
	{
		printf("Real Data\n");
		//readRealData("realworld/lena.tif_vertex.txt","realworld/lena.tif_constraint.txt",&triInput);
		readRealData_Binary("realworld/9.4M_vertex.txt","realworld/9.4M_constraint.txt",0,1000000,0, &triInput);
	}
	else
	{
		printf("Random Input\n");
		// points and distribution
		// 0:Uniform,1:Gaussian,2:Disk,3:Circle (4:Grid, 5:Ellipse, do not use)
		int distribution = 0;
		int seed = 0;
		double min_input_angle = 5.0;
		int numOfPoints = 100000;
		int numOfSegments = numOfPoints*0.5;

		printf("Number of points = %d, Number of segments = %d, Distribution = %d, seed = %d\n",numOfPoints,numOfSegments,distribution,seed);

		if(!readInput(&triInput,distribution,seed,numOfPoints,numOfSegments,min_input_angle))
		{
			time_t rawtime;
			struct tm * timeinfo;
			time ( &rawtime );
			timeinfo = localtime ( &rawtime );
			printf ( "Starting time and date: %s", asctime (timeinfo) );

			GenerateRandomInput(numOfPoints,numOfSegments,seed,distribution,&triInput,min_input_angle);
			saveInput(&triInput,distribution,seed,numOfPoints,numOfSegments,min_input_angle);
		}

		// resize to fit draw window
		if(draw)
		{
			for(int i=0; i< triInput.numberofpoints; i++)
			{
				triInput.pointlist[2*i] = triInput.pointlist[2*i]*750/(triInput.numberofpoints*10);
				triInput.pointlist[2*i+1] = triInput.pointlist[2*i+1]*750/(triInput.numberofpoints*10);
			}
		}
	}

	printf("\nInput information:\n");
	printf("Number of points = %d\nnumber of triangles = %d\nnumber of segments = %d\n",
		triInput.numberofpoints,triInput.numberoftriangles,triInput.numberofsegments);
	printf("\n");

	// 1. Compute Refinement
	printf("--------------------------------------------------------\n");
	printf("1. Computing Refinement\n");
	if(!run_cpu && !run_gpu)
		printf("N/A\n");
	else
		printf("\n");

	// timer
	StopWatchInterface *timer = 0; 
    sdkCreateTimer( &timer );

	// runtime
	double cpu_time, gpu_time;

	// result
	triangulateio cpu_result,gpu_result;

	// 1.1 On CPU
	if(run_cpu)
	{
		printf("1.1 Computing Refinement on CPU\n");
		sdkResetTimer( &timer );
		sdkStartTimer( &timer );
		
		CPU_Triangle_Quality(&triInput,&cpu_result,theta,mode);

		sdkStopTimer( &timer );
		cpu_time = sdkGetTimerValue( &timer );
		printf("Total time = %.3f ms\n", cpu_time);
		printf("Number of points = %d\nnumber of triangles = %d\nnumber of segments = %d\n",
			cpu_result.numberofpoints,cpu_result.numberoftriangles,cpu_result.numberofsegments);
		printf("\n");

		// Check result
		if(check_result)
		{
			printf("Checking result... ");
			if(checkResult(&triInput,&gpu_result,theta))
				printf("PASS all tests!\n");
			else
				printf("Fail!\n");
		}
	}
	
	// 1.2 On GPU
	if(run_gpu)
	{
		printf("1.2 Computing Refinement on GPU\n");
		sdkResetTimer( &timer );
		sdkStartTimer( &timer );

		GPU_Refine_Quality(&triInput,&gpu_result,theta,insertpolicy,deletepolicy,mode,-1,
			&debug_ps,&debug_ts);

		sdkStopTimer( &timer );

		gpu_time = sdkGetTimerValue( &timer );
		printf("Total time = %.3f ms\n", gpu_time);
		printf("Number of points = %d\nnumberof triangles = %d\nnumber of segments = %d\n",
			gpu_result.numberofpoints,gpu_result.numberoftriangles,gpu_result.numberofsegments);
		printf("\n");

		// Check result
		if(check_result)
		{
			printf("Checking result... ");
			if(checkResult(&triInput,&gpu_result,theta))
				printf("PASS all tests!\n");
			else
				printf("Fail!\n");
		}
	}

	// 2. Compare results
	printf("--------------------------------------------------------\n");
	printf("2. Compare results on CPU and GPU\n");
	if(run_cpu && run_gpu)
	{
		printf("Time speedup = %.3f\n",cpu_time/gpu_time);
		printf("Points ratio = %.3f\n",gpu_result.numberofpoints*1.0/cpu_result.numberofpoints);
		printf("Triangles ratio = %.3f\n",gpu_result.numberoftriangles*1.0/cpu_result.numberoftriangles);
		printf("Segments ratio = %.3f\n",gpu_result.numberofsegments*1.0/cpu_result.numberofsegments);
	}
	else
		printf("N/A\n");

	// 3. Draw
	printf("--------------------------------------------------------\n");
	printf("3. Draw result\n");
	if(draw)
	{
		if(draw == 1)
		{
			printf("Draw result created by Triangle\n");
			drawTriangulation(argc,argv,&cpu_result);
		}
		else
		{
			printf("Draw result created by gQM\n");
			drawTriangulation(argc,argv,&gpu_result);
		}
	}
	else
		printf("N/A\n");
}