#ifndef _B_H
#define _B_H

#include <opencv2\objdetect\objdetect.hpp>
#include <opencv2\core\core.hpp>

#define CC_CASCADE_PARAMS "cascadeParams"
#define CC_STAGE_TYPE     "stageType"
#define CC_FEATURE_TYPE   "featureType"
#define CC_HEIGHT         "height"
#define CC_WIDTH          "width"

#define CC_STAGE_NUM    "stageNum"
#define CC_STAGES       "stages"
#define CC_STAGE_PARAMS "stageParams"

#define CC_BOOST            "BOOST"
#define CC_MAX_DEPTH        "maxDepth"
#define CC_WEAK_COUNT       "maxWeakCount"
#define CC_STAGE_THRESHOLD  "stageThreshold"
#define CC_WEAK_CLASSIFIERS "weakClassifiers"
#define CC_INTERNAL_NODES   "internalNodes"
#define CC_LEAF_VALUES      "leafValues"

#define CC_FEATURES       "features"
#define CC_FEATURE_PARAMS "featureParams"
#define CC_MAX_CAT_COUNT  "maxCatCount"

#define CC_HAAR   "HAAR"
#define CC_RECTS  "rects"
#define CC_TILTED "tilted"

#define CC_LBP  "LBP"
#define CC_RECT "rect"

#define CC_HOG  "HOG"

#define CV_SUM_PTRS( p0, p1, p2, p3, sum, rect, step )                    \
	/* (x, y) */                                                          \
	(p0) = sum + (rect).x + (step)* (rect).y, \
	/* (x + w, y) */                                                      \
	(p1) = sum + (rect).x + (rect).width + (step)* (rect).y, \
	/* (x + w, y) */                                                      \
	(p2) = sum + (rect).x + (step)* ((rect).y + (rect).height), \
	/* (x + w, y + h) */                                                  \
	(p3) = sum + (rect).x + (rect).width + (step)* ((rect).y + (rect).height)

#define CV_TILTED_PTRS( p0, p1, p2, p3, tilted, rect, step )                        \
	/* (x, y) */                                                                    \
	(p0) = tilted + (rect).x + (step)* (rect).y, \
	/* (x - h, y + h) */                                                            \
	(p1) = tilted + (rect).x - (rect).height + (step)* ((rect).y + (rect).height), \
	/* (x + w, y + w) */                                                            \
	(p2) = tilted + (rect).x + (rect).width + (step)* ((rect).y + (rect).width), \
	/* (x + w - h, y + w + h) */                                                    \
	(p3) = tilted + (rect).x + (rect).width - (rect).height                         \
	+ (step)* ((rect).y + (rect).width + (rect).height)

#define CALC_SUM_(p0, p1, p2, p3, offset) \
	((p0)[offset] - (p1)[offset] - (p2)[offset] + (p3)[offset])

#define CALC_SUM(rect,offset) CALC_SUM_((rect)[0], (rect)[1], (rect)[2], (rect)[3], offset)

using namespace cv;
class LBPEvaluator : public FeatureEvaluator
{
public:
	struct Feature
	{
		Feature();
		Feature(int x, int y, int _block_w, int _block_h) :
			rect(x, y, _block_w, _block_h) {}

		int calc(int offset) const;
		void updatePtrs(const Mat& sum);
		bool read(const FileNode& node);

		Rect rect; // weight and height for block
		const int* p[16]; // fast
	};

	LBPEvaluator();
	virtual ~LBPEvaluator();

	bool read(const FileNode& node);
	//virtual Ptr<FeatureEvaluator> clone() const;
	//virtual int getFeatureType() const { return FeatureEvaluator::LBP; }

	bool setImage(const Mat& image, Size _origWinSize);
	//virtual bool setWindow(Point pt);

	/*int operator()(int featureIdx) const
	{
		return featuresPtr[featureIdx].calc(offset);
	}*/
	/*virtual int calcCat(int featureIdx) const
	{
		return featuresPtr[featureIdx].calc(offset);
	}*/

	Size origWinSize;
	Ptr<vector<Feature> > features;
	Feature* featuresPtr; // optimization
	//Mat sum0, sum;
	//Rect normrect;

	int offset;
};
class CasBoost 
{
public:
	CasBoost();
	CasBoost(const std::string& filename);
	virtual ~CasBoost();
	virtual bool empty() const;
	bool load(const std::string& filename);
	virtual bool read(const cv::FileNode& node);
	
	bool setImage(const cv::Mat&);
	int predict(const Mat& img);


protected:
	enum { BOOST = 0 };
	enum {
		DO_CANNY_PRUNING = 1, SCALE_IMAGE = 2,
		FIND_BIGGEST_OBJECT = 4, DO_ROUGH_SEARCH = 8
	};


	class Data
	{
	public:
		struct  DTreeNode
		{
			int featureIdx;
			float threshold; // for ordered features only
			int left;
			int right;
		};

		struct  DTree
		{
			int nodeCount;
		};

		struct  Stage
		{
			int first;
			int ntrees;
			float threshold;
		};

		bool read(const cv::FileNode &node);

		bool isStumpBased;

		int stageType;
		int featureType;
		int ncategories;
		cv::Size origWinSize;

		std::vector<Stage> stages;
		std::vector<DTree> classifiers;
		std::vector<DTreeNode> nodes;
		std::vector<float> leaves;
		std::vector<int> subsets;
	};

	Data data;
	Ptr<LBPEvaluator> featureEvaluator;
	Ptr<CvHaarClassifierCascade> oldCascade;

};

class HaarEvaluator 
{
public:
	struct Feature
	{
		Feature();

		float calc(int offset) const;
		//inline void updatePtrs(const Mat& sum);
		bool read(const FileNode& node);

		bool tilted;

		enum { RECT_NUM = 3 };

		struct
		{
			Rect r;
			float weight;
		} rect[RECT_NUM];

		const int* p[RECT_NUM][4];
	};

	HaarEvaluator();
	virtual ~HaarEvaluator();

	virtual bool read(const FileNode& node);
	virtual bool setImage(const Mat&, Size _origWinSize);

	Size origWinSize;
	Ptr<vector<Feature> > features;
	Feature* featuresPtr; // optimization
	bool hasTiltedFeatures;

	Mat sum0, sqsum0, tilted0;
	Mat sum, sqsum, tilted;

	Rect normrect;
	const int *p[4];
	const double *pq[4];

	int offset;
	double varianceNormFactor;
};

inline HaarEvaluator::Feature::Feature()
{
	tilted = false;
	rect[0].r = rect[1].r = rect[2].r = Rect();
	rect[0].weight = rect[1].weight = rect[2].weight = 0;
	p[0][0] = p[0][1] = p[0][2] = p[0][3] =
		p[1][0] = p[1][1] = p[1][2] = p[1][3] =
		p[2][0] = p[2][1] = p[2][2] = p[2][3] = 0;
}

inline float HaarEvaluator::Feature::calc(int _offset) const
{
	float ret = rect[0].weight * CALC_SUM(p[0], _offset) + rect[1].weight * CALC_SUM(p[1], _offset);

	if (rect[2].weight != 0.0f)
		ret += rect[2].weight * CALC_SUM(p[2], _offset);

	return ret;
}



inline LBPEvaluator::Feature::Feature()
{
	rect = Rect();
	for (int i = 0; i < 16; i++)
		p[i] = 0;
}

inline int LBPEvaluator::Feature::calc(int _offset) const
{
	int s = CALC_SUM_(p[0], p[3], p[12], p[15],_offset);
	int s2 = CALC_SUM_(p[5], p[6], p[9], p[10], _offset);
	float cval = (s - s2) / 8.0;
	return (CALC_SUM_(p[0], p[1], p[4], p[5], _offset) >= cval ? 128 : 0) |   // 0
		(CALC_SUM_(p[1], p[2], p[5], p[6], _offset) >= cval ? 64 : 0) |    // 1
		(CALC_SUM_(p[2], p[3], p[6], p[7], _offset) >= cval ? 32 : 0) |    // 2
		(CALC_SUM_(p[6], p[7], p[10], p[11], _offset) >= cval ? 16 : 0) |  // 5
		(CALC_SUM_(p[10], p[11], p[14], p[15], _offset) >= cval ? 8 : 0) |  // 8
		(CALC_SUM_(p[9], p[10], p[13], p[14], _offset) >= cval ? 4 : 0) |   // 7
		(CALC_SUM_(p[8], p[9], p[12], p[13], _offset) >= cval ? 2 : 0) |    // 6
		(CALC_SUM_(p[4], p[5], p[8], p[9], _offset) >= cval ? 1 : 0);
}

inline void LBPEvaluator::Feature::updatePtrs(const Mat& _sum)
{
	const int* ptr = (const int*)_sum.data;
	size_t step = _sum.step / sizeof(ptr[0]);
	Rect tr = rect;
	CV_SUM_PTRS(p[0], p[1], p[4], p[5], ptr, tr, step);
	tr.x += 2 * rect.width;
	CV_SUM_PTRS(p[2], p[3], p[6], p[7], ptr, tr, step);
	tr.y += 2 * rect.height;
	CV_SUM_PTRS(p[10], p[11], p[14], p[15], ptr, tr, step);
	tr.x -= 2 * rect.width;
	CV_SUM_PTRS(p[8], p[9], p[12], p[13], ptr, tr, step);
}

#endif