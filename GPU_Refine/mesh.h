#ifndef MESH_H
#define MESH_H

#include <math.h>
#include <vector>
#include "triangle.h"
#include <set>
//#include <GL\glut.h>

// Constant number
#define REAL double
#define REAL2 double2

#define PI 3.141592653589793238462643383279502884197169399375105820974944592308

// Some useful structures
typedef struct vertex
{
    REAL x, y;	
	vertex(void){}
	vertex(REAL a, REAL b){x = a; y = b;}

} triVertex;

struct CompareByPosition {
    bool operator()(const triVertex &lhs, const triVertex &rhs) const {
        if (lhs.x != rhs.x) 
            return lhs.x < rhs.x;
        return lhs.y < rhs.y;
    }
};

typedef struct edge
{
	int x, y;
	edge(void){}
	edge(int a, int b){x = a; y = b;}
} triEdge;

struct CompareByEndPoints {
    bool operator()(const triEdge &lhs, const triEdge &rhs) const {
		return ( (lhs.x == rhs.x && lhs.y == rhs.y) || (lhs.x == rhs.y && lhs.y == rhs.x) );
    }
};

// Random number generator, obtained from http://oldmill.uchicago.edu/~wilder/Code/random/

void randinit(unsigned long x_);

unsigned long znew(); 

unsigned long wnew(); 

unsigned long MWC();

unsigned long SHR3();

unsigned long CONG(); 

unsigned long rand_int();         // [0,2^32-1]

float random();     // [0,1]

void GaussianRand(float *x, float *y);

// Generate a random DT mesh (Continuous Space)
void GenerateRandomInput(int numOfPoints, int numOfSegments ,int seed, int distribution,triangulateio *result, double min_input_angle);

void createTriangluation(triangulateio * input,char * name);

void createPSLG(triangulateio * input,char * name);

void saveCDT(triangulateio * input,char * name);

bool readCDT(triangulateio * input,char * name);

#endif