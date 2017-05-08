#include "m.h"




inline void _bitset(unsigned long * a, unsigned long b)
{
	*a |= 1 << b;
}
inline void _bitreset(unsigned long * a, unsigned long b)
{
	*a &= ~(1 << b);
}



void initERComp(ERConnectedComp* comp)
{
	comp->size = 0;
	comp->history = NULL;
	comp->min_x = 0x0fffffff;
	comp->min_y = 0x0fffffff;
	comp->max_x = 0;
	comp->max_y = 0;
}


void ERNewHistory(ERConnectedComp* comp, ERGrowHistory** last_node_ERs)
{
	if (NULL == comp->history)
	{
		ERGrowHistory* history = (ERGrowHistory*)malloc(sizeof(ERGrowHistory));
		comp->history = history;
		history->next = last_node_ERs[comp->grey_level];
		last_node_ERs[comp->grey_level] = history;
		comp->history->val = comp->grey_level;
		history->flag = 0;

	}
	else if (comp->history->val != comp->grey_level)
	{
		ERGrowHistory* history = (ERGrowHistory*)malloc(sizeof(ERGrowHistory));
		
		comp->history->parent = history;

		comp->history = history;
		history->next=last_node_ERs[comp->grey_level];
		last_node_ERs[comp->grey_level] = history;
		comp->history->val = comp->grey_level;
		history->flag = 0;
	}
	comp->history->start_pos = comp->head;
	comp->history->end_pos = comp->tail;
	comp->history->size = comp->size;
	comp->history->max_x = comp->max_x;
	comp->history->max_y = comp->max_y;
	comp->history->min_x = comp->min_x;
	comp->history->min_y = comp->min_y;
}


void ERMergeComp(ERConnectedComp* comp1,
ERConnectedComp* comp2,
ERConnectedComp* comp,
ERGrowHistory** last_node_ERs)
{
	LinkedPoint* head;
	LinkedPoint* tail;
	ERNewHistory(comp1, last_node_ERs);
	ERNewHistory(comp2, last_node_ERs);

	comp1->history->parent = comp2->history;
	if (comp1->size > 0 && comp2->size > 0)
	{
		comp1->tail->next = comp2->head;
		//comp2->head->prev = comp1->tail;
	}
	head = (comp1->size > 0) ? comp1->head : comp2->head;
	tail = (comp2->size > 0) ? comp2->tail : comp1->tail;
	comp->head = head;
	comp->tail = tail;
	comp->grey_level = comp2->grey_level;
	comp->history = comp2->history;
	comp->size = comp1->size + comp2->size;
	/*comp->history->start_pos =comp->head;
	comp->history->end_pos = comp->tail;*/
	comp->max_x = (comp1->max_x > comp2->max_x) ? comp1->max_x : comp2->max_x;
	comp->max_y = (comp1->max_y > comp2->max_y) ? comp1->max_y : comp2->max_y;
	comp->min_x = (comp1->min_x < comp2->min_x) ? comp1->min_x : comp2->min_x;
	comp->min_y = (comp1->min_y < comp2->min_y) ? comp1->min_y : comp2->min_y;
}

void accumulateERComp(ERConnectedComp* comp, LinkedPoint* point)
{
	if (comp->size > 0)
	{
		//point->prev = comp->tail;
		comp->tail->next = point;
		point->next = NULL;
		if (point->pt.x > comp->max_x)
		{
			comp->max_x = point->pt.x;
		}
		if (point->pt.y > comp->max_y)
		{
			 comp->max_y=point->pt.y;
		}
		if (point->pt.x < comp->min_x)
		{
			comp->min_x = point->pt.x;
		}
		if (point->pt.y < comp->min_y)
		{
			comp->min_y = point->pt.y;
		}
	}
	else {
		//point->prev = NULL;
		point->next = NULL;
		comp->head = point;
		comp->min_x = point->pt.x;
		comp->min_y = point->pt.y;
		comp->max_x = point->pt.x;
		comp->max_y = point->pt.y;
	}
	comp->tail = point;
	comp->size++;
}

int* preprocessER_8UC1(CvMat* img,
	int*** heap_cur,
	CvMat* src,
	CvMat* mask)
{
	int srccpt = src->step - src->cols;
	int cpt_1 = img->cols - src->cols - 1;
	int* imgptr = img->data.i;
	int* startptr;

	int level_size[256];
	for (int i = 0; i < 256; i++)
		level_size[i] = 0;

	for (int i = 0; i < src->cols + 2; i++)
	{
		*imgptr = -1;
		imgptr++;
	}
	imgptr += cpt_1 - 1;
	uchar* srcptr = src->data.ptr;
	if (mask)
	{
		startptr = 0;
		uchar* maskptr = mask->data.ptr;
		for (int i = 0; i < src->rows; i++)
		{
			*imgptr = -1;
			imgptr++;
			for (int j = 0; j < src->cols; j++)
			{
				if (*maskptr)
				{
					if (!startptr)
						startptr = imgptr;

					*srcptr = 0xff - *srcptr;  //a

					level_size[*srcptr]++;
					*imgptr = ((*srcptr >> 5) << 8) | (*srcptr);
				}
				else {
					*imgptr = -1;
				}
				imgptr++;
				srcptr++;
				maskptr++;
			}
			*imgptr = -1;
			imgptr += cpt_1;
			srcptr += srccpt;
			maskptr += srccpt;
		}
	}
	else {
		startptr = imgptr + img->cols + 1;
		for (int i = 0; i < src->rows; i++)
		{
			*imgptr = -1;
			imgptr++;
			for (int j = 0; j < src->cols; j++)
			{

				*srcptr = 0xff - *srcptr;  //a

				level_size[*srcptr]++;
				*imgptr = ((*srcptr >> 5) << 8) | (*srcptr);
				imgptr++;
				srcptr++;
			}
			*imgptr = -1;
			imgptr += cpt_1;
			srcptr += srccpt;
		}
	}
	for (int i = 0; i < src->cols + 2; i++)
	{
		*imgptr = -1;
		imgptr++;
	}

	heap_cur[0][0] = 0;
	for (int i = 1; i < 256; i++)
	{
		heap_cur[i] = heap_cur[i - 1] + level_size[i - 1] + 1;
		heap_cur[i][0] = 0;
	}
	return startptr;
}

void extractER_8UC1_Pass(int* ioptr,
	int* imgptr,
	int*** heap_cur,
	LinkedPoint* ptsptr,
	ERConnectedComp* comptr,
	int step,
	int stepmask,
	int stepgap,
	ERGrowHistory** last_node_ERs)
{
	comptr->grey_level = 256;
	comptr++;
	comptr->grey_level = (*imgptr) & 0xff;
	initERComp(comptr);
	*imgptr |= 0x80000000;
	heap_cur += (*imgptr) & 0xff;
	int dir[] = { 1, step, -1, -step };
#ifdef __INTRIN_ENABLED__
	unsigned long heapbit[] = { 0, 0, 0, 0, 0, 0, 0, 0 };
	unsigned long* bit_cur = heapbit + (((*imgptr) & 0x700) >> 8);
#endif
	for (;;)
	{
		// take tour of all the 4 directions
		while (((*imgptr) & 0x70000) < 0x40000)
		{
			// get the neighbor
			int* imgptr_nbr = imgptr + dir[((*imgptr) & 0x70000) >> 16];
			if (*imgptr_nbr >= 0) // if the neighbor is not visited yet
			{
				*imgptr_nbr |= 0x80000000; // mark it as visited
				if (((*imgptr_nbr) & 0xff) < ((*imgptr) & 0xff))
				{
					// when the value of neighbor smaller than current
					// push current to boundary heap and make the neighbor to be the current one
					// create an empty comp
					(*heap_cur)++;
					**heap_cur = imgptr;
					*imgptr += 0x10000;
					heap_cur += ((*imgptr_nbr) & 0xff) - ((*imgptr) & 0xff);
#ifdef __INTRIN_ENABLED__
					_bitset(bit_cur, (*imgptr) & 0x1f);
					bit_cur += (((*imgptr_nbr) & 0x700) - ((*imgptr) & 0x700)) >> 8;
#endif
					imgptr = imgptr_nbr;
					comptr++;
					initERComp(comptr);
					comptr->grey_level = (*imgptr) & 0xff;
					continue;
				}
				else {
					// otherwise, push the neighbor to boundary heap
					heap_cur[((*imgptr_nbr) & 0xff) - ((*imgptr) & 0xff)]++;
					*heap_cur[((*imgptr_nbr) & 0xff) - ((*imgptr) & 0xff)] = imgptr_nbr;
#ifdef __INTRIN_ENABLED__
					_bitset(bit_cur + ((((*imgptr_nbr) & 0x700) - ((*imgptr) & 0x700)) >> 8), (*imgptr_nbr) & 0x1f);
#endif
				}
			}
			*imgptr += 0x10000;
		}
		int imsk = (int)(imgptr - ioptr);
		ptsptr->pt = cvPoint(imsk&stepmask, imsk >> stepgap);
		// get the current location
		accumulateERComp(comptr, ptsptr);
		ptsptr++;
		// get the next pixel from boundary heap
		if (**heap_cur)
		{
			imgptr = **heap_cur;
			(*heap_cur)--;
#ifdef __INTRIN_ENABLED__
			if (!**heap_cur)
				_bitreset(bit_cur, (*imgptr) & 0x1f);
#endif
		}
		else {
#ifdef __INTRIN_ENABLED__
			bool found_pixel = 0;
			unsigned long pixel_val;
			for (int i = ((*imgptr) & 0x700) >> 8; i < 8; i++)
			{
				if (_BitScanForward(&pixel_val, *bit_cur))
				{
					found_pixel = 1;
					pixel_val += i << 5;
					heap_cur += pixel_val - ((*imgptr) & 0xff);
					break;
				}
				bit_cur++;
			}
			if (found_pixel)
#else
			heap_cur++;
			unsigned long pixel_val = 0;
			for (unsigned long i = ((*imgptr) & 0xff) + 1; i < 256; i++)
			{
				if (**heap_cur)
				{
					pixel_val = i;
					break;
				}
				heap_cur++;
			}
			if (pixel_val)
#endif
			{
				imgptr = **heap_cur;
				(*heap_cur)--;
#ifdef __INTRIN_ENABLED__
				if (!**heap_cur)
					_bitreset(bit_cur, pixel_val & 0x1f);
#endif
				if (pixel_val < comptr[-1].grey_level)
				{
					
					ERNewHistory(comptr, last_node_ERs);
					comptr[0].grey_level = pixel_val;
				}
				else {
					// keep merging top two comp in stack until the grey level >= pixel_val
					for (;;)
					{
						comptr--;
						ERMergeComp(comptr + 1, comptr, comptr,last_node_ERs);
						if (pixel_val <= comptr[0].grey_level)
							break;
						if (pixel_val < comptr[-1].grey_level)
						{
							ERNewHistory(comptr, last_node_ERs);
							comptr[0].grey_level = pixel_val;
							break;
						}
					}
				}
			}
			else
			{
				ERNewHistory(comptr, last_node_ERs);
				comptr->history->parent = comptr->history;
				break;
			}
		}
	}
}


void extractER_8UC1(CvMat* src,
	CvMat* mask,
	ERGrowHistory** last_node_ERs,
	LinkedPoint* pts)
{
	int step = 8;
	int stepgap = 3;
	while (step < src->step + 2)
	{
		step <<= 1;
		stepgap++;
	}
	int stepmask = step - 1;

	// to speedup the process, make the width to be 2^N
	CvMat* img = cvCreateMat(src->rows + 2, step, CV_32SC1);
	int* ioptr = img->data.i + step + 1;
	int* imgptr;
	
	// pre-allocate boundary heap
	int** heap = (int**)cvAlloc((src->rows*src->cols + 256)*sizeof(heap[0]));
	int** heap_start[256];
	heap_start[0] = heap;

	
	ERConnectedComp comp[257];

	// darker to brighter (ER-)
	imgptr = preprocessER_8UC1(img, heap_start, src, mask);
	extractER_8UC1_Pass(ioptr, imgptr, heap_start, pts, comp, step, stepmask, stepgap, last_node_ERs);

	// brighter to darker (ER+)
	/*imgptr = preprocessER_8UC1(img, heap_start, src, mask);
	extractER_8UC1_Pass(ioptr, imgptr, heap_start, pts, history, comp, step, stepmask, stepgap,  node_last_array);*/

	// clean up
	//cvFree(&history);
	cvFree(&heap);
	//cvFree(&pts);
	cvReleaseMat(&img);
}

void
extractER(CvArr* _img,
CvArr* _mask,
ERGrowHistory** last_node_ERs,
LinkedPoint* pts)
{
	CvMat srchdr, *src = cvGetMat(_img, &srchdr);
	CvMat maskhdr, *mask = _mask ? cvGetMat(_mask, &maskhdr) : 0;

	CV_Assert(src != 0);
	CV_Assert(CV_MAT_TYPE(src->type) == CV_8UC1 || CV_MAT_TYPE(src->type) == CV_8UC3);
	CV_Assert(mask == 0 || (CV_ARE_SIZES_EQ(src, mask) && CV_MAT_TYPE(mask->type) == CV_8UC1));


	extractER_8UC1(src, mask, last_node_ERs, pts);
}