#ifndef _M_H
#define _M_H

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/core/internal.hpp"

#include <algorithm>

#ifdef HAVE_TEGRA_OPTIMIZATION
#include "opencv2/features2d/features2d_tegra.hpp"
#endif


typedef struct LinkedPoint
{
	//struct LinkedPoint* prev;
	struct LinkedPoint* next;
	cv::Point pt;
}
LinkedPoint;

// the history of region grown
typedef struct ERGrowHistory
{
	struct ERGrowHistory* parent;
	struct ERGrowHistory* next;
	LinkedPoint* start_pos; 
	LinkedPoint* end_pos;
	int size;
	int val;
	int flag;
	int max_x;
	int max_y;
	int min_x;
	int min_y;
}
ERGrowHistory;

typedef struct ERConnectedComp
{
	LinkedPoint* head;
	LinkedPoint* tail;
	ERGrowHistory* history;
	int grey_level;
	int size;
	int max_x;
	int max_y;
	int min_x;
	int min_y;
}
ERConnectedComp;

typedef struct ERList
{
	ERGrowHistory* ER_node;
	ERList* next;
}ERList;


void extractER(CvArr* _img,CvArr* _mask,
ERGrowHistory** last_node_ERs,LinkedPoint* pts);

#endif