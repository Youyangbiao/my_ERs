#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\objdetect\objdetect.hpp>
#include "b.h"
#include "m.h"
#include <time.h>
#include <math.h>

#define ER_area(node) ( (node->max_x-node->min_x+1)*(node->max_y-node->min_y+1) )

using namespace cv;

inline void free_list(ERList* header)
{
	ERList* node;
	while (header)
	{
		node = header;
		header = header->next;
		free(node);
	}
}

void test5(Mat& src, const Mat& img, const Mat& yrb, CasBoost& strong_casBoost, CasBoost& weak_casBoost, int& pos_code,
	ERList* strong_char_header, ERList* weak_char_header)
{
	LinkedPoint* pts = (LinkedPoint*)malloc(img.rows*img.cols*sizeof(LinkedPoint));
	ERGrowHistory** last_node_ERs = (ERGrowHistory**)malloc(256 * sizeof(ERGrowHistory*));
	memset(last_node_ERs, 0, 256 * sizeof(ERGrowHistory*));
	extractER(&((CvMat)img), NULL, last_node_ERs, pts);
	Mat show_img;
	ERGrowHistory* node = NULL, *c_node = NULL, *p_node = NULL, *stab_node = NULL, *pp_node = NULL;
	int depth = 1;

	float stab = 10000, s = 0;

	Mat part_img,sum_img;
	char* win = (char*)malloc(15);
	int r[11] = { 0 };
	float ER_width = 0, ER_height = 0;
	float aspect_ratio;
	int n = 0;

	ERList* strong_char = NULL, *weak_char = NULL;
	LinkedPoint* l_point = NULL;
	int y_avg = 0, cr_avg = 0, cb_avg = 0;
	for (int i = 0; i < 249; i++)
	{
		node = last_node_ERs[i];
		while (node)
		{
			if (node->flag == 0)
			{
				depth = 1;
				c_node = node;
				p_node = c_node->parent;
				c_node->flag = 1;
				//while (((float)c_node->size / p_node->size)>0.7  && c_node != p_node)
				while (((float)ER_area(c_node) / ER_area(p_node))>0.7  && p_node->parent != p_node)
				{
					//c_node = p_node;
					p_node->flag = 1;
					p_node = p_node->parent;
					//c_node->flag = 1;
					depth++;
				}
				pp_node = p_node;
				if (depth > 3)
				{
					c_node = node;
					p_node = c_node->parent->parent;
					depth = depth - 2;
					stab = 10000;
					while (depth)
					{
						s = (ER_area(p_node) - ER_area(c_node)) / (float)ER_area(c_node);
						//s = (p_node->size - c_node->size) / (float)(c_node->size);
						if (s <= stab)
						{
							stab = s;
							stab_node = c_node;
						}
						c_node = c_node->parent;
						p_node = p_node->parent;
						depth--;
					}


					ER_width = stab_node->max_x - stab_node->min_x + 1;
					ER_height = stab_node->max_y - stab_node->min_y + 1;
					aspect_ratio = ER_width / ER_height;
					if (stab_node->size<(img.cols*img.rows / 4) && aspect_ratio>0.2&&aspect_ratio < 3)
					{
						show_img = img(Rect(stab_node->min_x, stab_node->min_y, (int)ER_width, (int)ER_height));
						resize(show_img, part_img, Size(24, 24));
						integral(part_img, sum_img);
						int result = strong_casBoost.predict(sum_img);
						if (result <= -5)
						{
							n++;
							int weak_result = weak_casBoost.predict(sum_img);
							if (weak_result>0)//
							{
								weak_char = (ERList*)malloc(sizeof(ERList));
								weak_char->next = weak_char_header->next;
								weak_char_header->next = weak_char;
								weak_char->min_x = stab_node->min_x;
								weak_char->min_y = stab_node->min_y;
								weak_char->max_x = stab_node->max_x;
								weak_char->max_y = stab_node->max_y;
								weak_char->width = ER_width;
								weak_char->height = ER_height;
								l_point = stab_node->start_pos;
								y_avg = 0;
								cr_avg = 0;
								cb_avg = 0;
								while (l_point != stab_node->end_pos)
								{
									y_avg += yrb.at<Vec3b>(l_point->pt)[0];
									cr_avg += yrb.at<Vec3b>(l_point->pt)[1];
									cb_avg += yrb.at<Vec3b>(l_point->pt)[2];
									l_point = l_point->next;
								}
								y_avg += yrb.at<Vec3b>(l_point->pt)[0];
								cr_avg += yrb.at<Vec3b>(l_point->pt)[1];
								cb_avg += yrb.at<Vec3b>(l_point->pt)[2];
								y_avg /= stab_node->size;
								cr_avg /= stab_node->size;
								cb_avg /= stab_node->size;
								weak_char->y_avg = y_avg;
								weak_char->cr_avg = cr_avg;
								weak_char->cb_avg = cb_avg;
							}
						}
						if (result>0)
						{
							r[10]++;
							if (1)
							{
								strong_char = (ERList*)malloc(sizeof(ERList));
								strong_char->next = strong_char_header->next;
								strong_char_header->next = strong_char;
								strong_char->min_x = stab_node->min_x;
								strong_char->min_y = stab_node->min_y;
								strong_char->max_x = stab_node->max_x;
								strong_char->max_y = stab_node->max_y;
								strong_char->width = ER_width;
								strong_char->height = ER_height;
								l_point = stab_node->start_pos;
								y_avg = 0;
								cr_avg = 0;
								cb_avg = 0;
								while (l_point != stab_node->end_pos)
								{
									y_avg += yrb.at<Vec3b>(l_point->pt)[0];
									cr_avg += yrb.at<Vec3b>(l_point->pt)[1];
									cb_avg += yrb.at<Vec3b>(l_point->pt)[2];
									l_point = l_point->next;
								}
								y_avg += yrb.at<Vec3b>(l_point->pt)[0];
								cr_avg += yrb.at<Vec3b>(l_point->pt)[1];
								cb_avg += yrb.at<Vec3b>(l_point->pt)[2];
								y_avg /= stab_node->size;
								cr_avg /= stab_node->size;
								cb_avg /= stab_node->size;
								strong_char->y_avg = y_avg;
								strong_char->cr_avg = cr_avg;
								strong_char->cb_avg = cb_avg;
							}
							p_node = pp_node;
							while (p_node->parent != p_node)
							{
								p_node->flag = 1;
								p_node = p_node->parent;
							}
							pos_code++;
						}
						else
						{
							r[-result]++;
						}
					}
				}
			}
			node = node->next;
		}
	}


	for (int i = 0; i < 10; i++)
	{
		printf("stage %d:%d\n", i, r[i]);
	}

	printf("neg(stage5-9):%d\n", n);
	printf("pos:%d\n", r[10]);
	printf("\n");

	for (int i = 0; i < 256; i++)
	{
		while (last_node_ERs[i])
		{
			node = last_node_ERs[i];
			last_node_ERs[i] = last_node_ERs[i]->next;
			free(node);
		}
	}
	free(last_node_ERs);
	free(pts);
	free(win);
}

void func6()
{
	char* img_path = (char*)malloc(45);
	Mat img, src;
	CasBoost weak_casBoost(".\\data_FA0.5\\cascade.xml");
	CasBoost strong_casBoost(".\\data_FA0.15\\cascade.xml");
	int pos_code, neg_code;
	pos_code = 0;
	neg_code = 0;
	int pic_code = 100;
	sprintf(img_path, "E:\\ref-字符识别\\icdar2013\\%d.jpg", pic_code);
	//sprintf(img_path, "E:\\ref-字符识别\\icdar2013_test\\img_201.jpg");
	src = imread(img_path);
	//cvtColor(src, img, CV_BGR2GRAY);
	ERList* strong_char_header = (ERList*)malloc(sizeof(ERList));
	ERList* weak_char_header = (ERList*)malloc(sizeof(ERList));
	strong_char_header->next = NULL;
	weak_char_header->next = NULL;
	Mat yrb;
	cvtColor(src, yrb, CV_BGR2YCrCb);

	img = imread(img_path, 0);
	test5(src, img, yrb, strong_casBoost, weak_casBoost, pos_code, strong_char_header, weak_char_header);
	test5(src, img, yrb, strong_casBoost, weak_casBoost, pos_code, strong_char_header, weak_char_header);

	ERList* strong_tail = NULL, *strong_char = NULL;
	ERList* weak_char_pre = NULL, *weak_char = NULL;
	strong_char = strong_char_header;
	while (strong_char->next)
	{
		strong_char = strong_char->next;
	}
	strong_tail = strong_char;
	strong_char = strong_char_header->next;
	int dis_thresold, dis;
	int width_threshold, height_threshold;
	int n = 0;
	char* win = new char[8];
	while (strong_char)
	{
		weak_char_pre = weak_char_header;
		weak_char = weak_char_pre->next;
		dis_thresold = strong_char->height > strong_char->width ? strong_char->height*strong_char->height : strong_char->width*strong_char->width;
		dis_thresold *= 4;
		while (weak_char)
		{
			dis = (strong_char->min_x - weak_char->min_x)*(strong_char->min_x - weak_char->min_x) +
				(strong_char->min_y - weak_char->min_y)*(strong_char->min_y - weak_char->min_y);
			if (dis < dis_thresold)
			{
				width_threshold = strong_char->width < weak_char->width ? strong_char->width : weak_char->width;
				height_threshold = strong_char->height < weak_char->height ? strong_char->height : strong_char->width;
				if ((abs(strong_char->width - weak_char->width) < width_threshold) && (abs(strong_char->height - weak_char->height) < height_threshold))
				{
					if ((abs(strong_char->y_avg - weak_char->y_avg) < 25) && (abs(strong_char->cr_avg - weak_char->cr_avg) < 25) &&
						(abs(strong_char->cb_avg - weak_char->cb_avg) < 25))
					{
						weak_char_pre->next = weak_char->next;
						strong_tail->next = weak_char;
						strong_tail = weak_char;
						strong_tail->next = NULL;
						weak_char = weak_char_pre->next;
					}
					else
					{
						weak_char_pre = weak_char_pre->next;
						weak_char = weak_char->next;
					}
				}
				else
				{
					weak_char_pre = weak_char_pre->next;
					weak_char = weak_char->next;
				}
			}
			else
			{
				weak_char_pre = weak_char_pre->next;
				weak_char = weak_char->next;
			}
		}
		if (n < pos_code)
		{
			sprintf(win, "%d", n);
			rectangle(src, Rect(strong_char->min_x, strong_char->min_y, (int)strong_char->width, (int)strong_char->height), Scalar(0, 255, 0), 1, 8);
			putText(src, win, Point(strong_char->min_x, strong_char->min_y - 2), FONT_HERSHEY_SIMPLEX, 0.3, Scalar(0, 255, 0));
		}
		else
		{
			sprintf(win, "%d", n);
			rectangle(src, Rect(strong_char->min_x, strong_char->min_y, (int)strong_char->width, (int)strong_char->height), Scalar(0, 0, 255), 1, 8);
			putText(src, win, Point(strong_char->min_x, strong_char->min_y - 2), FONT_HERSHEY_SIMPLEX, 0.3, Scalar(0, 0, 255));
		}
		n++;
		strong_char = strong_char->next;
	}
	delete win;
	free_list(strong_char_header);
	free_list(weak_char_header);
	namedWindow("B", WINDOW_NORMAL);
	free(img_path);
	cv::imshow("B", src);
	cv::waitKey(0);
}

void func7()
{
	char* img_path = (char*)malloc(45);
	Mat img, src;
	CasBoost weak_casBoost(".\\data_FA0.5\\cascade.xml");
	CasBoost strong_casBoost(".\\data_FA0.15\\cascade.xml");
	int pos_code, neg_code;
	pos_code = 0;
	neg_code = 0;
	int pic_code = 100;
	sprintf(img_path, "E:\\ref-字符识别\\icdar2013\\%d.jpg", pic_code);
	//sprintf(img_path, "E:\\ref-字符识别\\icdar2013_test\\img_201.jpg");
	src = imread(img_path);
	//cvtColor(src, img, CV_BGR2GRAY);
	ERList* strong_char_header = (ERList*)malloc(sizeof(ERList));
	ERList* weak_char_header = (ERList*)malloc(sizeof(ERList));
	strong_char_header->next = NULL;
	weak_char_header->next = NULL;
	Mat yrb;
	cvtColor(src, yrb, CV_BGR2YCrCb);

	img = imread(img_path, 0);
	test5(src, img, yrb, strong_casBoost, weak_casBoost, pos_code, strong_char_header, weak_char_header);
	test5(src, img, yrb, strong_casBoost, weak_casBoost, pos_code, strong_char_header, weak_char_header);

	ERList* strong_tail = NULL, *strong_char = NULL;
	ERList* weak_char_pre = NULL, *weak_char = NULL;
	strong_char = strong_char_header;
	while (strong_char->next)
	{
		strong_char = strong_char->next;
	}
	strong_tail = strong_char;
	strong_char = strong_char_header->next;
	int dis_thresold, dis;
	int width_threshold, height_threshold;
	int n = 0;
	char* win = new char[8];
	while (strong_char)
	{
		weak_char_pre = weak_char_header;
		weak_char = weak_char_pre->next;
		dis_thresold = strong_char->height > strong_char->width ? strong_char->height*strong_char->height : strong_char->width*strong_char->width;
		dis_thresold *= 4;
		while (weak_char)
		{
			dis = (strong_char->min_x - weak_char->min_x)*(strong_char->min_x - weak_char->min_x) +
				(strong_char->min_y - weak_char->min_y)*(strong_char->min_y - weak_char->min_y);
			if (dis < dis_thresold)
			{
				width_threshold = strong_char->width < weak_char->width ? strong_char->width : weak_char->width;
				height_threshold = strong_char->height < weak_char->height ? strong_char->height : strong_char->width;
				if ((abs(strong_char->width - weak_char->width) < width_threshold) && (abs(strong_char->height - weak_char->height) < height_threshold))
				{
					if ((abs(strong_char->y_avg - weak_char->y_avg) < 25) && (abs(strong_char->cr_avg - weak_char->cr_avg) < 25) &&
						(abs(strong_char->cb_avg - weak_char->cb_avg) < 25))
					{
						weak_char_pre->next = weak_char->next;
						strong_tail->next = weak_char;
						strong_tail = weak_char;
						strong_tail->next = NULL;
						weak_char = weak_char_pre->next;
					}
					else
					{
						weak_char_pre = weak_char_pre->next;
						weak_char = weak_char->next;
					}
				}
				else
				{
					weak_char_pre = weak_char_pre->next;
					weak_char = weak_char->next;
				}
			}
			else
			{
				weak_char_pre = weak_char_pre->next;
				weak_char = weak_char->next;
			}
		}
		n++;
		strong_char = strong_char->next;
	}
	ERList* strong_char_pre = strong_char_header;
	strong_char = strong_char_pre->next;
	int overlap_width, overlap_height;
	float IOU,IOU2;
	int char_area, char_area2;
	bool clear_flag=false;
	while (strong_char)
	{
		ERList* strong_char2_pre = strong_char;
		ERList* strong_char2 = strong_char2_pre->next;
		while (strong_char2)
		{
			overlap_width = (strong_char->max_x<strong_char2->max_x ? strong_char->max_x : strong_char2->max_x) -
				(strong_char->min_x>strong_char2->min_x ? strong_char->min_x : strong_char2->min_x);
			overlap_height = (strong_char->max_y<strong_char2->max_y ? strong_char->max_y : strong_char2->max_y) -
				(strong_char->min_y>strong_char2->min_y ? strong_char->min_y : strong_char2->min_y);
			if (overlap_height <= 0 || overlap_width <= 0)
			{
				strong_char2 = strong_char2->next;
				strong_char2_pre = strong_char2_pre->next;
			}
			else
			{
				char_area = strong_char->height*strong_char->width;
				char_area2 = strong_char2->height*strong_char2->width;
				//IOU = (float)(overlap_height*overlap_width) / (char_area + char_area2 - overlap_height*overlap_width);
				IOU = (float)(overlap_height*overlap_width) / (char_area);
				IOU2 = (float)(overlap_height*overlap_width) / (char_area2);
				if (IOU > 0.5||IOU2>0.5)
				{
					if (char_area2 > char_area)
					{
						clear_flag = true;
						break;
					}
					else
					{
						strong_char2_pre->next = strong_char2->next;
						free(strong_char2);
						strong_char2 = strong_char2_pre->next;
					}
				}
				else
				{
					strong_char2 = strong_char2->next;
					strong_char2_pre = strong_char2_pre->next;
				}
			}
		}
		if (clear_flag)
		{
			strong_char_pre->next = strong_char->next;
			free(strong_char);
			strong_char = strong_char_pre->next;
			clear_flag = false;
		}
		else
		{
			strong_char = strong_char->next;
			strong_char_pre = strong_char_pre->next;
		}
	}
	strong_char = strong_char_header->next;
	while (strong_char)
	{
		sprintf(win, "%d", n);
		rectangle(src, Rect(strong_char->min_x, strong_char->min_y, (int)strong_char->width, (int)strong_char->height), Scalar(0, 255, 0), 1, 8);
		putText(src, win, Point(strong_char->min_x, strong_char->min_y - 2), FONT_HERSHEY_SIMPLEX, 0.3, Scalar(0, 255, 0));
		strong_char = strong_char->next;
	}

	delete win;
	free_list(strong_char_header);
	free_list(weak_char_header);
	namedWindow("B", WINDOW_NORMAL);
	free(img_path);
	cv::imshow("B", src);
	cv::waitKey(0);
}

void func8()
{
	char* img_path = (char*)malloc(45);
	Mat img, src;
	CasBoost weak_casBoost(".\\data_FA0.5\\cascade.xml");
	CasBoost strong_casBoost(".\\data_FA0.15\\cascade.xml");
	int pos_code, neg_code;
	pos_code = 0;
	neg_code = 0;
	int pic_code = 166;
	sprintf(img_path, "E:\\ref-字符识别\\icdar2013\\%d.jpg", pic_code);
	//sprintf(img_path, "E:\\ref-字符识别\\icdar2013_test\\img_201.jpg");
	src = imread(img_path);
	//cvtColor(src, img, CV_BGR2GRAY);
	ERList* strong_char_header = (ERList*)malloc(sizeof(ERList));
	ERList* weak_char_header = (ERList*)malloc(sizeof(ERList));
	strong_char_header->next = NULL;
	weak_char_header->next = NULL;
	Mat yrb;
	cvtColor(src, yrb, CV_BGR2YCrCb);
	vector<Mat> channels;
	split(yrb, channels);

	//img = imread(img_path, 0);
	test5(src, channels[0], yrb, strong_casBoost, weak_casBoost, pos_code, strong_char_header, weak_char_header);
	test5(src, channels[0], yrb, strong_casBoost, weak_casBoost, pos_code, strong_char_header, weak_char_header);
	test5(src, channels[1], yrb, strong_casBoost, weak_casBoost, pos_code, strong_char_header, weak_char_header);
	test5(src, channels[1], yrb, strong_casBoost, weak_casBoost, pos_code, strong_char_header, weak_char_header);
	test5(src, channels[2], yrb, strong_casBoost, weak_casBoost, pos_code, strong_char_header, weak_char_header);
	test5(src, channels[2], yrb, strong_casBoost, weak_casBoost, pos_code, strong_char_header, weak_char_header);

	ERList* strong_tail = NULL, *strong_char = NULL;
	ERList* weak_char_pre = NULL, *weak_char = NULL;
	strong_char = strong_char_header;
	while (strong_char->next)
	{
		strong_char = strong_char->next;
	}
	strong_tail = strong_char;
	strong_char = strong_char_header->next;
	int dis_thresold, dis;
	int width_threshold, height_threshold;
	int n = 0;
	char* win = new char[8];
	while (strong_char)
	{
		weak_char_pre = weak_char_header;
		weak_char = weak_char_pre->next;
		dis_thresold = strong_char->height > strong_char->width ? strong_char->height*strong_char->height : strong_char->width*strong_char->width;
		dis_thresold *= 4;
		while (weak_char)
		{
			dis = (strong_char->min_x - weak_char->min_x)*(strong_char->min_x - weak_char->min_x) +
				(strong_char->min_y - weak_char->min_y)*(strong_char->min_y - weak_char->min_y);
			if (dis < dis_thresold)
			{
				width_threshold = strong_char->width < weak_char->width ? strong_char->width : weak_char->width;
				height_threshold = strong_char->height < weak_char->height ? strong_char->height : strong_char->width;
				if ((abs(strong_char->width - weak_char->width) < width_threshold) && (abs(strong_char->height - weak_char->height) < height_threshold))
				{
					if ((abs(strong_char->y_avg - weak_char->y_avg) < 25) && (abs(strong_char->cr_avg - weak_char->cr_avg) < 25) &&
						(abs(strong_char->cb_avg - weak_char->cb_avg) < 25))
					{
						weak_char_pre->next = weak_char->next;
						strong_tail->next = weak_char;
						strong_tail = weak_char;
						strong_tail->next = NULL;
						weak_char = weak_char_pre->next;
					}
					else
					{
						weak_char_pre = weak_char_pre->next;
						weak_char = weak_char->next;
					}
				}
				else
				{
					weak_char_pre = weak_char_pre->next;
					weak_char = weak_char->next;
				}
			}
			else
			{
				weak_char_pre = weak_char_pre->next;
				weak_char = weak_char->next;
			}
		}
		n++;
		strong_char = strong_char->next;
	}
	ERList* strong_char_pre = strong_char_header;
	strong_char = strong_char_pre->next;
	int overlap_width, overlap_height;
	float IOU, IOU2;
	int char_area, char_area2;
	bool clear_flag = false;
	while (strong_char)
	{
		ERList* strong_char2_pre = strong_char;
		ERList* strong_char2 = strong_char2_pre->next;
		while (strong_char2)
		{
			overlap_width = (strong_char->max_x<strong_char2->max_x ? strong_char->max_x : strong_char2->max_x) -
				(strong_char->min_x>strong_char2->min_x ? strong_char->min_x : strong_char2->min_x);
			overlap_height = (strong_char->max_y<strong_char2->max_y ? strong_char->max_y : strong_char2->max_y) -
				(strong_char->min_y>strong_char2->min_y ? strong_char->min_y : strong_char2->min_y);
			if (overlap_height <= 0 || overlap_width <= 0)
			{
				strong_char2 = strong_char2->next;
				strong_char2_pre = strong_char2_pre->next;
			}
			else
			{
				char_area = strong_char->height*strong_char->width;
				char_area2 = strong_char2->height*strong_char2->width;
				//IOU = (float)(overlap_height*overlap_width) / (char_area + char_area2 - overlap_height*overlap_width);
				IOU = (float)(overlap_height*overlap_width) / (char_area);
				IOU2 = (float)(overlap_height*overlap_width) / (char_area2);
				if (IOU > 0.5 || IOU2>0.5)
				{
					if (char_area2 > char_area)
					{
						clear_flag = true;
						break;
					}
					else
					{
						strong_char2_pre->next = strong_char2->next;
						free(strong_char2);
						strong_char2 = strong_char2_pre->next;
					}
				}
				else
				{
					strong_char2 = strong_char2->next;
					strong_char2_pre = strong_char2_pre->next;
				}
			}
		}
		if (clear_flag)
		{
			strong_char_pre->next = strong_char->next;
			free(strong_char);
			strong_char = strong_char_pre->next;
			clear_flag = false;
		}
		else
		{
			strong_char = strong_char->next;
			strong_char_pre = strong_char_pre->next;
		}
	}
	strong_char = strong_char_header->next;
	int cnt = 0;
	while (strong_char)
	{
		sprintf(win, "%d", cnt);
		rectangle(src, Rect(strong_char->min_x, strong_char->min_y, (int)strong_char->width, (int)strong_char->height), Scalar(0, 255, 0), 1, 8);
		putText(src, win, Point(strong_char->min_x, strong_char->min_y - 2), FONT_HERSHEY_SIMPLEX, 0.3, Scalar(0, 255, 0));
		strong_char = strong_char->next;
		cnt++;
	}

	delete win;
	free_list(strong_char_header);
	free_list(weak_char_header);
	namedWindow("B", WINDOW_NORMAL);
	free(img_path);
	cv::imshow("B", src);
	cv::waitKey(0);
}

int main()
{
	func8();
	/*Mat src = imread("E:\\RM\\test\\0165.jpg");
	Mat img;
	cvtColor(src, img, CV_BGR2GRAY);
	Mat model = imread("E:\\RM\\test\\1835_50.jpg",0);
	Point minloc, maxloc;
	double minval, maxval;
	Mat result;
	matchTemplate(img, model, result, CV_TM_SQDIFF);
	minMaxLoc(result, &minval, &maxval, &minloc, &maxloc, Mat());
	rectangle(src, Rect(minloc.x, minloc.y, 22, 26), Scalar(0, 255, 0), 1, 8);
	imshow("pic", src);
	waitKey(0);*/
}