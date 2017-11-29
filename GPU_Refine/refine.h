#ifndef REFINE_H
#define REFINE_H

#include <cuda_runtime.h>
#include <helper_timer.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/partition.h>
#include <thrust/scan.h>
#include <thrust/remove.h>
#include <thrust/unique.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/iterator/reverse_iterator.h>
#include "triangle.h"


#define BLOCK_SIZE 256

#define REAL double
#define REAL2 double2
#define MAKE_REAL2  make_double2
#define BYTE char
#define uint64	unsigned long long

#define MAXINT 2147483647
#define PI 3.141592653589793238462643383279502884197169399375105820974944592308

#define LS 50

typedef enum {Circumcenter, Offcenter} InsertPolicy;
typedef enum {Strict, Connected, Orthogonal, noDeletion, DeleteFromDT, noPriority} DeletePolicy;
typedef enum {DoMark, DoVote, DoCount, DoCollect} CollectMode;

typedef struct PStatus
{
	unsigned char _status; 

	// 76543210
	//    |||||----- isDeleted
	//    ||||------ isSteiner
	//    |||------- isSegmentSplit
	//    ||-------- isNew (i.e. created this round)
	//    |--------- isRedundant (i.e. need to be deleted)

	__forceinline__ __host__ __device__
	PStatus() : _status(0){}

	__forceinline__ __device__ __host__
	void setDeleted()		{ _status = ( _status | (1 << 0) ); }

	__forceinline__ __device__ __host__
	bool isDeleted() const	{ return (_status & (1 << 0)) > 0; }

	__forceinline__ __device__ __host__
	void setSteiner()		{ _status = ( _status | (1 << 1) ); }

	__forceinline__ __device__ __host__
	bool isSteiner() const	{ return (_status & (1 << 1)) > 0; }

	__forceinline__ __device__ __host__
	void setSegmentSplit()		{ _status = ( _status | (1 << 2) ); }

	__forceinline__ __device__ __host__
	void setTriangleSplit()	{ _status = ( _status & ~(1 << 2) ); }

	__forceinline__ __device__ __host__
	bool isSegmentSplit() const	{ return (_status & (1 << 2)) > 0; }

	__forceinline__ __device__ __host__
	void setNew()			{ _status = ( _status | (1 << 3) ); }

	__forceinline__ __device__ __host__
	void setOld()			{ _status = ( _status & ~(1 << 3) ); }

	__forceinline__ __device__ __host__
	bool isNew()			{ return (_status & (1 << 3)) > 0; }

	__forceinline__ __device__ __host__
	void setReduntant()		{ _status = ( _status | (1 << 4) ); }

	__forceinline__ __device__ __host__
	bool isReduntant() const { return (_status & (1 << 4)) > 0; }

	__forceinline__ __device__ __host__
	void createNewSegmentSplit()
	{
		_status = 0; 

		setSteiner(); 
		setNew(); 
		setSegmentSplit(); 
	}

	__forceinline__ __device__ __host__
	void createNewTriangleSplit()
	{
		_status = 0; 

		setSteiner(); 
		setNew(); 
		setTriangleSplit(); 
	}

} PStatus;

typedef struct TStatus
{
	unsigned char _status; 

	// 76543210
	// ||||||||----- triangle status
	// |||||||------ bad
	// ||||||------- check
	// |||||-------- flipflop marker
	// ||||--------- flipOri

	__forceinline__ __host__ __device__ 
	TStatus(void)
	{
		_status = 0;
	}
	
	__forceinline__ __host__ __device__ 
	TStatus(bool n, bool b, bool c)
	{
		_status = 0;

		setNull(n); 
		setCheck(c);
		setBad(b); 
	}

	//status of triangles. 0 = null; 1 = not null
	__forceinline__ __host__ __device__ 
	void setNull(bool n) 
	{
		if(n)
			_status = _status & (~1);
		else
			_status = _status | 1;  
	}

	__forceinline__ __host__ __device__ 
	bool isNull() const
	{
		return ( _status & 1) == 0; 
	}

	__forceinline__ __host__ __device__ 
	void clear()
	{
		_status = 0; // clear all information
	}

	__forceinline__ __host__ __device__ 
	void setBad(bool bad)
	{
		_status = ( _status & ~(1 << 1) ) | (bad ? 1 : 0) << 1; 
	}

	__forceinline__ __host__ __device__ 
	bool isBad() const
	{
		return ( _status & (1 << 1) ) > 0; 
	}

	__forceinline__ __host__ __device__ 
	void setCheck(bool check)
	{
		_status = ( _status & ~(1 << 2) ) | (check ? 1 : 0) << 2; 
	}

	__forceinline__ __host__ __device__ 
	bool isCheck() const
	{
		return ( _status & (1 << 2) ) > 0; 
	}

	__forceinline__ __host__ __device__ 
	void setFlip(bool flip)
	{
		_status = ( _status & ~(1 << 3) ) | (flip ? 1 : 0) << 3; 
	}

	__forceinline__ __host__ __device__ 
	bool isFlip() const
	{
		return ( _status & (1 << 3) ) > 0; 
	}

	__forceinline__ __host__ __device__ 
	void setFlipOri(int ori)
	{
		_status = ( _status & ~(15 << 4) ) | (ori << 4); 
	}

	__forceinline__ __host__ __device__ 
	int getFlipOri() const
	{
		return ( _status & (15 << 4) ) >> 4; 
	}

}TStatus;

typedef struct MyList
{
	int a[LS];
	int size;

	__forceinline__ __host__ __device__ 
	MyList(void)
	{
		size = 0;
	}

	__forceinline__ __host__ __device__ 
	bool push(int v)
	{
		if(size == LS)
			return false;
		a[size] = v;
		size++;
		return true;
	}

	__forceinline__ __host__ __device__
	bool pop(void)
	{
		if(size == 0)
			return false;
		else
			size--;
	}

	__forceinline__ __host__ __device__
	bool find(int v)
	{
		bool result = false;
		for(int i=0; i<size; i++)
		{
			if(a[i] == v)
			{
				result = true;
				break;
			}
		}
		return result;
	}

	__forceinline__ __host__ __device__
	int getSize(void)
	{
		return size;
	}

	__forceinline__ __host__ __device__
	int getLast(void)
	{
		return a[size-1];
	}

}MyList;

struct isEmpty
{
	__device__ bool operator()( const TStatus t ) 
	{
		return t.isNull();	
	}
};

struct isDeleted
{
	__device__ bool operator()( const PStatus x ) 
	{
		return x.isDeleted();	
	}
};

struct isCheckOrBadTriangle
{
	__device__ bool operator()( const TStatus t ) 
	{
		return ( !t.isNull() && ( t.isCheck() || t.isBad() ) ); 
	}
};

struct isBadTriangle
{
	__device__ bool operator()( const TStatus t ) 
	{
		return ( !t.isNull() && t.isBad() ); 
	}
};

struct isFlipFlop
{
	__device__ bool operator()( const TStatus t ) 
	{
		return ( !t.isNull() && t.isFlip() ); 
	}
};

struct isFlipWinner
{
	__device__ bool operator()( const TStatus t ) 
	{
		return ( !t.isNull() && (t.getFlipOri() != 15) ); 
	}
};

struct isNotNegativeInt
{
	__device__ bool operator()( const int x ) 
	{
		return !( x < 0 ); 
	}
};

struct isNegativeInt
{
	__device__ bool operator()( const int x ) 
	{
		return x < 0; 
	}
};

struct isNotNegative
{
	__device__ bool operator()( const BYTE x ) 
	{
		return !( x < 0 ); 
	}
};

struct isTriangleSplit
{
	__device__ bool operator()( const PStatus x ) 
	{
		return ( !x.isDeleted() && x.isSteiner() && !x.isSegmentSplit());  
	}
};

struct isRedundant
{
	__device__ bool operator()( const PStatus x ) 
	{
		return ( !x.isDeleted() && x.isReduntant());
	}
};

// Thrust
typedef thrust::device_ptr<int> IntDPtr; 
typedef thrust::device_ptr<char> ByteDPtr; 
typedef thrust::device_ptr<unsigned long long> UInt64DPtr; 
typedef thrust::device_ptr<int2> Int2DPtr; 
typedef thrust::device_ptr<REAL2> Real2DPtr;

typedef thrust::device_vector<int> IntD;
typedef thrust::device_vector<int2> Int2D;
typedef thrust::device_vector<BYTE> ByteD;
typedef thrust::device_vector<unsigned long long> UInt64D; 
typedef thrust::device_vector<REAL2> Real2D;
typedef thrust::device_vector<REAL> RealD;

typedef thrust::host_vector<int> IntH;
typedef thrust::host_vector<REAL2> Real2H;
typedef thrust::host_vector<REAL> RealH;

typedef thrust::device_vector<PStatus>PStatusD;
typedef thrust::device_ptr<PStatus>PStatusDPtr;
typedef thrust::host_vector<PStatus>PStatusH;

typedef thrust::device_vector<TStatus>TStatusD;
typedef thrust::device_ptr<TStatus>TStatusDPtr;
typedef thrust::host_vector<TStatus>TStatusH;

typedef thrust::device_vector<int2>IntPairD;

typedef thrust::device_vector<unsigned long long>ULongLongD;


void GPU_Refine_Quality(triangulateio *input, triangulateio *result, double theta, InsertPolicy insertpolicy,DeletePolicy deletepolicy, int mode,int debug_iter,
	PStatus **ps_debug,TStatus **ts_debug);

#endif