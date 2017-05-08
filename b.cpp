#include "b.h"
#include <opencv\cv.h>


CasBoost::CasBoost()
{

}
CasBoost::~CasBoost()
{

}

CasBoost::CasBoost(const std::string& filename)
{
	load(filename);
}

bool CasBoost::empty() const
{
	return oldCascade.empty() && data.stages.empty();
}

bool CasBoost::Data::read(const FileNode &root)
{
	static const float THRESHOLD_EPS = 1e-5f;

	// load stage params
	string stageTypeStr = (string)root[CC_STAGE_TYPE];
	if (stageTypeStr == CC_BOOST)
		stageType = BOOST;
	else
		return false;

	string featureTypeStr = (string)root[CC_FEATURE_TYPE];
	if (featureTypeStr == CC_HAAR)
		featureType = FeatureEvaluator::HAAR;
	else if (featureTypeStr == CC_LBP)
		featureType = FeatureEvaluator::LBP;
	else if (featureTypeStr == CC_HOG)
		featureType = FeatureEvaluator::HOG;

	else
		return false;

	origWinSize.width = (int)root[CC_WIDTH];
	origWinSize.height = (int)root[CC_HEIGHT];
	CV_Assert(origWinSize.height > 0 && origWinSize.width > 0);

	isStumpBased = (int)(root[CC_STAGE_PARAMS][CC_MAX_DEPTH]) == 1 ? true : false;

	// load feature params
	FileNode fn = root[CC_FEATURE_PARAMS];
	if (fn.empty())
		return false;

	ncategories = fn[CC_MAX_CAT_COUNT];
	int subsetSize = (ncategories + 31) / 32,
		nodeStep = 3 + (ncategories>0 ? subsetSize : 1);

	// load stages
	fn = root[CC_STAGES];
	if (fn.empty())
		return false;

	stages.reserve(fn.size());
	classifiers.clear();
	nodes.clear();

	FileNodeIterator it = fn.begin(), it_end = fn.end();

	for (int si = 0; it != it_end; si++, ++it)
	{
		FileNode fns = *it;
		Stage stage;
		stage.threshold = (float)fns[CC_STAGE_THRESHOLD] - THRESHOLD_EPS;
		fns = fns[CC_WEAK_CLASSIFIERS];
		if (fns.empty())
			return false;
		stage.ntrees = (int)fns.size();
		stage.first = (int)classifiers.size();
		stages.push_back(stage);
		classifiers.reserve(stages[si].first + stages[si].ntrees);

		FileNodeIterator it1 = fns.begin(), it1_end = fns.end();
		for (; it1 != it1_end; ++it1) // weak trees
		{
			FileNode fnw = *it1;
			FileNode internalNodes = fnw[CC_INTERNAL_NODES];
			FileNode leafValues = fnw[CC_LEAF_VALUES];
			if (internalNodes.empty() || leafValues.empty())
				return false;

			DTree tree;
			tree.nodeCount = (int)internalNodes.size() / nodeStep;
			classifiers.push_back(tree);

			nodes.reserve(nodes.size() + tree.nodeCount);
			leaves.reserve(leaves.size() + leafValues.size());
			if (subsetSize > 0)
				subsets.reserve(subsets.size() + tree.nodeCount*subsetSize);

			FileNodeIterator internalNodesIter = internalNodes.begin(), internalNodesEnd = internalNodes.end();

			for (; internalNodesIter != internalNodesEnd;) // nodes
			{
				DTreeNode node;
				node.left = (int)*internalNodesIter; ++internalNodesIter;
				node.right = (int)*internalNodesIter; ++internalNodesIter;
				node.featureIdx = (int)*internalNodesIter; ++internalNodesIter;
				if (subsetSize > 0)
				{
					for (int j = 0; j < subsetSize; j++, ++internalNodesIter)
						subsets.push_back((int)*internalNodesIter);
					node.threshold = 0.f;
				}
				else
				{
					node.threshold = (float)*internalNodesIter; ++internalNodesIter;
				}
				nodes.push_back(node);
			}

			internalNodesIter = leafValues.begin(), internalNodesEnd = leafValues.end();

			for (; internalNodesIter != internalNodesEnd; ++internalNodesIter) // leaves
				leaves.push_back((float)*internalNodesIter);
		}
	}

	return true;
}

bool CasBoost::load(const std::string& filename)
{
	oldCascade.release();
	data = Data();
	featureEvaluator.release();

	cv::FileStorage fs(filename, cv::FileStorage::READ);
	if (!fs.isOpened())
		return false;

	if (read(fs.getFirstTopLevelNode()))
		return true;

	fs.release();

	oldCascade = cv::Ptr<CvHaarClassifierCascade>((CvHaarClassifierCascade*)cvLoad(filename.c_str(), 0, 0, 0));
	return !oldCascade.empty();
}

bool CasBoost::read(const cv::FileNode& root)
{
	if (!data.read(root))
		return false;

	// load features
	//featureEvaluator =cv::FeatureEvaluator::create(data.featureType);
	//featureEvaluator = Ptr<HaarEvaluator>(new HaarEvaluator);
	featureEvaluator = Ptr<LBPEvaluator>(new LBPEvaluator);
	cv::FileNode fn = root[CC_FEATURES];
	if (fn.empty())
		return false;

	return featureEvaluator->read(fn); 
}

int CasBoost::predict(const Mat& sum_img)
{
	featureEvaluator->setImage(sum_img,data.origWinSize);


	/*int nodeOfs = 0, leafOfs = 0;
	float* cascadeLeaves = &data.leaves[0];
	CasBoost::Data::DTreeNode* cascadeNodes = &data.nodes[0];
	CasBoost::Data::Stage* cascadeStages = &data.stages[0];

	int nstages = (int)data.stages.size();
	double sum;
	for (int stageIdx = 0; stageIdx < nstages; stageIdx++)
	{
		CasBoost::Data::Stage& stage = cascadeStages[stageIdx];
		sum = 0.0;

		int ntrees = stage.ntrees;
		for (int i = 0; i < ntrees; i++, nodeOfs++, leafOfs += 2)
		{
			CasBoost::Data::DTreeNode& node = cascadeNodes[nodeOfs];
			double value = featureEvaluator->featuresPtr[node.featureIdx].calc(0);
			sum += cascadeLeaves[value < node.threshold ? leafOfs : leafOfs + 1];
		}

		if (sum < stage.threshold)
			return -stageIdx;
	}

	return 1;*/
	int nstages = (int)data.stages.size();
	int nodeOfs = 0, leafOfs = 0;
	//FEval& featureEvaluator = (FEval&)*_featureEvaluator;
	size_t subsetSize = (data.ncategories + 31) / 32;
	int* cascadeSubsets = &data.subsets[0];
	float* cascadeLeaves = &data.leaves[0];
	CasBoost::Data::DTreeNode* cascadeNodes = &data.nodes[0];
	CasBoost::Data::Stage* cascadeStages = &data.stages[0];
	double sum;
#ifdef HAVE_TEGRA_OPTIMIZATION
	float tmp = 0; // float accumulator -- float operations are quicker
#endif
	for (int si = 0; si < nstages; si++)
	{
		CasBoost::Data::Stage& stage = cascadeStages[si];
		int wi, ntrees = stage.ntrees;
#ifdef HAVE_TEGRA_OPTIMIZATION
		tmp = 0;
#else
		sum = 0;
#endif

		for (wi = 0; wi < ntrees; wi++)
		{
			CasBoost::Data::DTreeNode& node = cascadeNodes[nodeOfs];
			int c = featureEvaluator->featuresPtr[node.featureIdx].calc(0);
			const int* subset = &cascadeSubsets[nodeOfs*subsetSize];
#ifdef HAVE_TEGRA_OPTIMIZATION
			tmp += cascadeLeaves[subset[c >> 5] & (1 << (c & 31)) ? leafOfs : leafOfs + 1];
#else
			sum += cascadeLeaves[subset[c >> 5] & (1 << (c & 31)) ? leafOfs : leafOfs + 1];
#endif
			nodeOfs++;
			leafOfs += 2;
		}
#ifdef HAVE_TEGRA_OPTIMIZATION
		if (tmp < stage.threshold) {
			sum = (double)tmp;
			return -si;
		}
#else
		if (sum < stage.threshold)
			return -si;
#endif
	}

#ifdef HAVE_TEGRA_OPTIMIZATION
	sum = (double)tmp;
#endif

	return 1;
}

HaarEvaluator::HaarEvaluator()
{
	features = new vector<Feature>();
}
HaarEvaluator::~HaarEvaluator()
{
}

bool HaarEvaluator::read(const FileNode& node)
{
	features->resize(node.size());
	featuresPtr = &(*features)[0];
	FileNodeIterator it = node.begin(), it_end = node.end();
	hasTiltedFeatures = false;

	for (int i = 0; it != it_end; ++it, i++)
	{
		if (!featuresPtr[i].read(*it))
			return false;
		if (featuresPtr[i].tilted)
			hasTiltedFeatures = true;
	}
	return true;
}

bool HaarEvaluator::Feature::read(const FileNode& node)
{
	FileNode rnode = node[CC_RECTS];
	FileNodeIterator it = rnode.begin(), it_end = rnode.end();

	int ri;
	for (ri = 0; ri < RECT_NUM; ri++)
	{
		rect[ri].r = Rect();
		rect[ri].weight = 0.f;
	}

	for (ri = 0; it != it_end; ++it, ri++)
	{
		FileNodeIterator it2 = (*it).begin();
		it2 >> rect[ri].r.x >> rect[ri].r.y >>
			rect[ri].r.width >> rect[ri].r.height >> rect[ri].weight;
	}

	tilted = (int)node[CC_TILTED] != 0;
	return true;
}

bool  HaarEvaluator::setImage(const Mat& img, Size _origWinSize)
{
	origWinSize = _origWinSize;
	int rn = origWinSize.height + 1, cn = origWinSize.width + 1;
	normrect = Rect(1, 1, origWinSize.width - 2, origWinSize.height - 2);
	sum0.create(rn, cn, CV_32S);
	sqsum0.create(rn, cn, CV_64F);

	sum = Mat(rn, cn, CV_32S, sum0.data);
	sqsum = Mat(rn, cn, CV_64F, sqsum0.data);

	integral(img, sum, sqsum);

	const int* sdata = (const int*)sum.data;
	const double* sqdata = (const double*)sqsum.data;
	size_t sumStep = sum.step / sizeof(sdata[0]);
	size_t sqsumStep = sqsum.step / sizeof(sqdata[0]);

	CV_SUM_PTRS(p[0], p[1], p[2], p[3], sdata, normrect, sumStep);
	CV_SUM_PTRS(pq[0], pq[1], pq[2], pq[3], sqdata, normrect, sqsumStep);

	size_t fi, nfeatures = features->size();
	size_t step = sum.step / sizeof(sdata[0]);
	for (fi = 0; fi < nfeatures; fi++)
	{
		CV_SUM_PTRS(featuresPtr[fi].p[0][0], featuresPtr[fi].p[0][1], featuresPtr[fi].p[0][2], featuresPtr[fi].p[0][3], sdata, featuresPtr[fi].rect[0].r, step);
		CV_SUM_PTRS(featuresPtr[fi].p[1][0], featuresPtr[fi].p[1][1], featuresPtr[fi].p[1][2], featuresPtr[fi].p[1][3], sdata, featuresPtr[fi].rect[1].r, step);
		if (featuresPtr[fi].rect[2].weight)
			CV_SUM_PTRS(featuresPtr[fi].p[2][0], featuresPtr[fi].p[2][1], featuresPtr[fi].p[2][2], featuresPtr[fi].p[2][3], sdata, featuresPtr[fi].rect[2].r, step);

	}

	int valsum = CALC_SUM(p, 0);
	double valsqsum = CALC_SUM(pq, 0);
	double nf = (double)normrect.area() * valsqsum - (double)valsum * valsum;
	if (nf > 0.)
		nf = sqrt(nf);
	else
		nf = 1.;
	varianceNormFactor = 1. / nf;
	offset = 0;
	return true;
}

LBPEvaluator::LBPEvaluator()
{
	features = new vector<Feature>();
}
LBPEvaluator::~LBPEvaluator()
{
}

bool LBPEvaluator::read(const FileNode& node)
{
	features->resize(node.size());
	featuresPtr = &(*features)[0];
	FileNodeIterator it = node.begin(), it_end = node.end();
	for (int i = 0; it != it_end; ++it, i++)
	{
		if (!featuresPtr[i].read(*it))
			return false;
	}
	return true;
}
bool LBPEvaluator::Feature::read(const FileNode& node)
{
	FileNode rnode = node[CC_RECT];
	FileNodeIterator it = rnode.begin();
	it >> rect.x >> rect.y >> rect.width >> rect.height;
	return true;
}

bool LBPEvaluator::setImage(const Mat& sum,  Size _origWinSize)
{
	origWinSize = _origWinSize;

	size_t fi, nfeatures = features->size();

	for (fi = 0; fi < nfeatures; fi++)
		featuresPtr[fi].updatePtrs(sum);

	offset = 0;
	return true;
}

