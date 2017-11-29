#include <stdio.h>
#include <string>
#include <sstream>
#include "mesh.h"
#include "predicates.h"

unsigned long z, w, jsr, jcong; // Seeds

void randinit(unsigned long x_) 
{ z =x_; w = x_; jsr = x_; jcong = x_; }

unsigned long znew() 
{ return (z = 36969 * (z & 0xfffful) + (z >> 16)); }

unsigned long wnew() 
{ return (w = 18000 * (w & 0xfffful) + (w >> 16)); }

unsigned long MWC()  
{ return ((znew() << 16) + wnew()); }

unsigned long SHR3()
{ jsr ^= (jsr << 17); jsr ^= (jsr >> 13); return (jsr ^= (jsr << 5)); }

unsigned long CONG() 
{ return (jcong = 69069 * jcong + 1234567); }

unsigned long rand_int()         // [0,2^32-1]
{ return ((MWC() ^ CONG()) + SHR3()); }

float random()     // [0,1]
{ return ((float) rand_int() / (float(ULONG_MAX)+1)); }

void GaussianRand(float *x, float *y)
{
    float x1, x2, w;

    do {
        x1 = 2.0 * random() - 1.0;
        x2 = 2.0 * random() - 1.0;
        w = x1 * x1 + x2 * x2;
    } while ( w >= 1.0 );

    w = sqrt( (-2.0 * log( w ) ) / w );
    *x = x1 * w;
    *y = x2 * w;
}

// Generate a random CDT mesh (Continuous Space)
void GenerateRandomInput(int numOfPoints, int numOfSegments,int seed, int distribution,triangulateio *result, double min_input_angle)
{    
	printf("Generating random input mesh...\n");
    
	float x, y;
    randinit(seed);
	// Step 0. Try to read old file to speed up the algorithm
	printf("Try to read old input file to speed up the program...\n");
	triangulateio oldInput;
	bool otag = false;
	int shift = 1000;
	while(!otag && shift < numOfSegments)
	{
		std::ostringstream strs;
		if(min_input_angle >= 60.0)
			strs << "input/d" << distribution << "_s" << seed << "_p" << numOfPoints << "_c" << (numOfSegments - shift);
		else
			strs << "input/d" << distribution << "_s" << seed << "_p" << numOfPoints << "_c" << (numOfSegments - shift) << "_with_minimum_input_angle_" << min_input_angle;
		std::string fn = strs.str();
		char *com = new char[fn.length() + 1];
		strcpy(com, fn.c_str());
		otag = readCDT(&oldInput,com);
		if(otag)
			printf("Finish reading %s\n",com);
		shift += 1000;
	}

	// Step 1. Generate random input points
	triangulateio inPointsCont;
    memset(&inPointsCont, 0, sizeof(triangulateio));
	inPointsCont.numberofpoints   = numOfPoints;
    inPointsCont.pointlist        = new double[inPointsCont.numberofpoints * 2];

	if(otag)
		memcpy(inPointsCont.pointlist, oldInput.pointlist, sizeof(double)*2*numOfPoints);
	else
	{
		std::set<triVertex, CompareByPosition> pointset;
		int num = 0;
		int min = 0;
		int max = numOfPoints*10;

		while(num < inPointsCont.numberofpoints && !otag)
		{

			x = 0.0;
			y = 0.0;
			// Generate points randomly
			switch (distribution)
			{
				// UniformDistribution
				case 0:
				{
					x = random();
					y = random();
					x = min + (max - min) * x;
					y = min + (max - min) * y;
				}
				break;
				// GaussianDistribution
				case 1:
				{
					float x1, x2, w;
					float tx, ty; 

					do {
						do {
							x1 = 2.0 * random() - 1.0;
							x2 = 2.0 * random() - 1.0;
							w = x1 * x1 + x2 * x2;
						} while ( w >= 1.0 );

						w = sqrt( (-2.0 * log( w ) ) / w );
						tx = x1 * w;
						ty = x2 * w;
					} while ( tx < -3 || tx >= 3 || ty < -3 || ty >= 3 );

					x = min + (max - min) * ( (tx + 3.0) / 6.0 );
					y = min + (max - min) * ( (ty + 3.0) / 6.0 );
				}
				break;
				// DiskDistribution
				case 2:
				{
					float d;
					do
					{
						x = random() - 0.5; 
						y = random() - 0.5; 
						d = x * x + y * y;
					} while ( d > 0.45 * 0.45 );

					x += 0.5;
					y += 0.5;
					x = x * (max - min ) + min;
					y = y * (max - min ) + min;
				}
				break;
				// CircleDistribution
				case 3:
				{
					float d, a;
					d = random() * 0.04; 
					a = random() * 3.141592654 * 2; 

					x = ( 0.45 + d ) * cos( a ); 
					y = ( 0.45 + d ) * sin( a ); 

					x += 0.5;
					y += 0.5;
					x = x * (max - min) + min;
					y = y * (max - min) + min;
				}
				break;
				// GridDistribution
				case 4:
				{
					float v[2];

					for ( int i = 0; i < 2; ++i )
					{
						const float val  = random() * max;
						const float frac = val - floor( val );
						v[ i ] = ( frac < 0.5f ) ? floor( val ) : ceil( val );
					}

					x = v[0];
					y = v[1];					
				}
				break;
				// EllipseDistribution
				case 5:
					{ 			
						float a = random() * PI * 2; 
						float radius = 0.75;

						x = radius * cos( a ); 
						y = radius * sin( a ); 

						x = x * 1.0 / 3.0; 
						y = y * 2.0 / 3.0; 

						x += 0.5;
						y += 0.5;
						x = x * (max - min) + min;
						y = y * (max - min) + min;
					}
				break;   
			}

			// Adjust to bounds
			if ( x >= max ) x = max;
			if ( x <  min ) x = min;
			if ( y >= max ) y = max;
			if ( y <  min ) y = min;			

			triVertex vert( x, y ); 

			if ( pointset.find( vert ) == pointset.end() ) 
			{
				pointset.insert(vert);
				*(inPointsCont.pointlist + 2*num) = x;
				*(inPointsCont.pointlist + 2*num + 1) = y;
				++num; 
			}
		}
	}
	
	// Step 2. Triangulate the point set to get the convex hull
	triangulateio dtResult;
    memset(&dtResult, 0, sizeof(triangulateio));
	triangulate("zQnc", &inPointsCont, &dtResult, NULL); //DT mesh

	int convexsize = dtResult.numberofsegments; // the number of convex hull segments

	// Step 3. Generate segments randomly
	int num = 0;
	int max = numOfPoints-1;

	if(otag)
	{
		inPointsCont.numberofsegments = oldInput.numberofsegments;
		inPointsCont.segmentlist      = new int[numOfSegments < convexsize ? convexsize*2 : numOfSegments*2];
		memcpy(inPointsCont.segmentlist,oldInput.segmentlist,2*sizeof(int)*oldInput.numberofsegments);
		num = oldInput.numberofsegments - convexsize;
	}
	else
	{
		inPointsCont.numberofsegments = convexsize;
		inPointsCont.segmentlist      = new int[numOfSegments < convexsize ? convexsize*2 : numOfSegments*2];
		memcpy(inPointsCont.segmentlist,dtResult.segmentlist,2*sizeof(int)*convexsize); // copy convex hull to input
	}



	REAL goodAngle = cos(min_input_angle * PI/180.0);
	goodAngle *= goodAngle;
	
	printf("Current number of segments generated: \n");
	while( num < numOfSegments - convexsize)
	{
		bool pass = true; // pass checking or not
		// random pick two points
		int p0 = random()*max;
		int p1 = random()*max;
		if(p0 == p1)
			continue;

		// check the length of the segment (only for experiment)
		triVertex ev[2];
		ev[0].x = inPointsCont.pointlist[2*p0];
		ev[0].y = inPointsCont.pointlist[2*p0+1];
		ev[1].x = inPointsCont.pointlist[2*p1];
		ev[1].y = inPointsCont.pointlist[2*p1+1];

		double distance = (ev[0].x-ev[1].x)*(ev[0].x-ev[1].x) + 
				(ev[0].y - ev[1].y)*(ev[0].y-ev[1].y);

		if(distance > 1100*1100)
			continue;

		// check segments situation
		bool incidentL = false; // incident to left point
		bool incidentR = false;
		for(int i = 0; i<inPointsCont.numberofsegments; i++)
		{
			int index0 = inPointsCont.segmentlist[2*i];
			int index1 = inPointsCont.segmentlist[2*i+1];
			triVertex v[4];
			v[0].x = inPointsCont.pointlist[2*p0];
			v[0].y = inPointsCont.pointlist[2*p0+1];
			v[1].x = inPointsCont.pointlist[2*p1];
			v[1].y = inPointsCont.pointlist[2*p1+1];
			v[2].x = inPointsCont.pointlist[2*index0];
			v[2].y = inPointsCont.pointlist[2*index0+1];
			v[3].x = inPointsCont.pointlist[2*index1];
			v[3].y = inPointsCont.pointlist[2*index1+1];

			if( (p0 == index0 && p1 == index1) || (p0 == index1 && p1 == index0))
			{
				// this segment already exists
				pass = false;
				break;
			}
			else if ( p0 == index0 || p0 == index1 || p1 == index0 || p1 == index1)
			{
				//continue; // dont need to check when want to generate input with small angles

				// this segment is incident to other segments
				// need to check the angle between them

				if( p0 == index0 || p0 == index1)
					incidentL = true;
				else
					incidentR = true;

				//if( incidentL && incidentR) // this segment is incident to more than 1 segments
				//{
				//	// prevent to form holes
				//	pass = false;
				//	break;
				//}

				REAL dx[2], dy[2], edgelength[2];
				edgelength[0] = (v[0].x-v[1].x)*(v[0].x-v[1].x) + 
					(v[0].y - v[1].y)*(v[0].y-v[1].y);
				edgelength[1] = (v[2].x-v[3].x)*(v[2].x-v[3].x) + 
					(v[2].y - v[3].y)*(v[2].y-v[3].y);
				if( p0 == index0 )
				{
					dx[0] = v[1].x - v[0].x;
					dx[1] = v[3].x - v[2].x;
					dy[0] = v[1].y - v[0].y;
					dy[1] = v[3].y - v[2].y;
				}
				else if ( p0 == index1)
				{
					dx[0] = v[1].x - v[0].x;
					dx[1] = v[2].x - v[3].x;
					dy[0] = v[1].y - v[0].y;
					dy[1] = v[2].y - v[3].y;
				}
				else if ( p1 == index0)
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
				if( dotproduct < 0) //obtuse angle
					continue;
				else
				{
					REAL cossquare = dotproduct*dotproduct /(edgelength[0]*edgelength[1]);
					if(cossquare > goodAngle)
					{
						// the angle between two segment
						pass = false;
						break;
					}
					else
					{
						continue;
					}
				}
			}
			else
			{
				// check if segments intersect each other
				// look along p0 and p1 direction
				REAL det0 = counterclockwise(&v[0],&v[2],&v[1]);
				REAL det1 = counterclockwise(&v[0],&v[3],&v[1]);
				// look along index0 and index1 direction
				REAL det2 = counterclockwise(&v[2],&v[0],&v[3]);
				REAL det3 = counterclockwise(&v[2],&v[1],&v[3]);
				if(det0*det1<=0 && det2*det3<=0)
				{
					pass = false;
					break;
				}
				else
					continue;
			}
		}
		if(!pass) // fail to generate, try again
			continue;
		else
		{
			inPointsCont.segmentlist[2*inPointsCont.numberofsegments] = p0;
			inPointsCont.segmentlist[2*inPointsCont.numberofsegments+1] = p1;
			inPointsCont.numberofsegments++;
			num++;
			if( (num + convexsize) % 1000 == 0)
				printf("%d\n",num + convexsize);
		}
	}

	// Step 5
	triangulateio cdtResult;
    memset(&cdtResult, 0, sizeof(triangulateio));
	triangulate("pzQnc", &inPointsCont, &cdtResult, NULL); //CDT mesh

	memcpy(result, &cdtResult, sizeof(triangulateio));
}

void saveCDT(triangulateio * input, char * name)
{
	FILE *fp;
	fp = fopen(name, "w");

	if(fp == NULL)
		printf("File Error!\n");

	fprintf(fp, "%d\n", input->numberofpoints);
	fprintf(fp, "%d\n", input->numberoftriangles);
	fprintf(fp, "%d\n", input->numberofsegments);

	for(int i=0; i<input->numberofpoints; i++)
		fprintf(fp, "%lf %lf\n",input->pointlist[2*i],input->pointlist[2*i+1]);
	for(int i=0; i<input->numberoftriangles; i++)
		fprintf(fp, "%d %d %d\n",input->trianglelist[3*i],input->trianglelist[3*i+1],input->trianglelist[3*i+2]);
	for(int i=0; i<input->numberoftriangles; i++)
		fprintf(fp, "%d %d %d\n",input->neighborlist[3*i],input->neighborlist[3*i+1],input->neighborlist[3*i+2]);
	for(int i=0; i<input->numberofsegments; i++)
		fprintf(fp, "%d %d\n",input->segmentlist[2*i],input->segmentlist[2*i+1]);

	fclose(fp);
}

bool readCDT(triangulateio * input, char * name)
{
	FILE *fp;
	fp = fopen(name, "r");

	if(fp == NULL)
	{
		printf("Cannot find the input file\n");
		return false;
	}

	char buf[100];
	
	int ln = 0;

	while (fgets(buf, 100, fp) != NULL) {
		if(ln == 0)
		{
			if(sscanf(buf,"%d",&(input->numberofpoints)) != 1)
			{
				printf("Incorrect format\n");
				fclose(fp);
				return false;
			}
			else
				input->pointlist = new double[2*input->numberofpoints];
		}
		else if(ln == 1)
		{
			if(sscanf(buf,"%d",&(input->numberoftriangles)) != 1)
			{
				printf("Incorrect format\n");
				fclose(fp);
				return false;
			}
			else
			{
				input->trianglelist = new int[3*input->numberoftriangles];
				input->neighborlist = new int[3*input->numberoftriangles];
			}
		}
		else if(ln == 2)
		{
			if(sscanf(buf,"%d",&(input->numberofsegments)) != 1)
			{
				printf("Incorrect format\n");
				fclose(fp);
				return false;
			}
			else
				input->segmentlist = new int[2*input->numberofsegments];
		}
		else if(ln < input->numberofpoints + 3)
		{
			if(sscanf(buf,"%lf %lf",
				input->pointlist + 2*(ln-3),
				input->pointlist + 2*(ln-3)+1) != 2)
			{
				printf("Incorrect format\n");
				fclose(fp);
				return false;
			}
		}
		else if(ln < input->numberofpoints + input->numberoftriangles + 3)
		{
			if(sscanf(buf,"%d %d %d",
				input->trianglelist + 3*(ln-input->numberofpoints-3),
				input->trianglelist + 3*(ln-input->numberofpoints-3)+1,
				input->trianglelist + 3*(ln-input->numberofpoints-3)+2) != 3)
			{
				printf("Incorrect format\n");
				fclose(fp);
				return false;
			}
		}
		else if(ln < input->numberofpoints + 2*input->numberoftriangles + 3)
		{
			if(sscanf(buf,"%d %d %d",
				input->neighborlist + 3*(ln-input->numberofpoints-input->numberoftriangles-3),
				input->neighborlist + 3*(ln-input->numberofpoints-input->numberoftriangles-3)+1,
				input->neighborlist + 3*(ln-input->numberofpoints-input->numberoftriangles-3)+2) != 3)
			{
				printf("Incorrect format\n");
				fclose(fp);
				return false;
			}
		}
		else if(ln < input->numberofpoints + 2*input->numberoftriangles + input->numberofsegments + 3)
		{
			if(sscanf(buf,"%d %d",
				input->segmentlist + 2*(ln-input->numberofpoints-2*input->numberoftriangles-3),
				input->segmentlist + 2*(ln-input->numberofpoints-2*input->numberoftriangles-3)+1) != 2)
			{
				printf("Incorrect format\n");
				fclose(fp);
				return false;
			}
		}
		else
			break;
		ln++;
	}

	fclose(fp);
	printf("Succeed\n");
	return true;
}