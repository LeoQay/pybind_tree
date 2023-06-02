#ifndef TREE_REGRESSOR_HPP
#define TREE_REGRESSOR_HPP


#include <tuple>
#include <vector>
#include <map>

#include <Eigen/Eigen>


using FLOAT = double;
using ArgsType = Eigen::Map<Eigen::VectorX<int>>;

using MatrixFLOAT = Eigen::Map<Eigen::Matrix<FLOAT, -1, -1, Eigen::RowMajor>>;
using VectorFLOAT = Eigen::Map<Eigen::VectorX<FLOAT>>;


struct Params
{
    Params() : min_samples_split(2), max_depth(-1), min_impurity_decrease(0.0) {}

    void setMaxDepth(int value) { max_depth = value; }
    void setMinImpurityDecrease(FLOAT value) { min_impurity_decrease = value; }
    void setMinSamplesSplit(int value) { min_samples_split = value; }

    int min_samples_split;
    int max_depth;
    FLOAT min_impurity_decrease;
};


struct Pair
{
    Pair() : feature_id(-1), threshold(0), left_count(0) {}

    Pair(int fet, FLOAT thr, int left) : feature_id(fet), threshold(thr), left_count(left) {}

    int feature_id;
    FLOAT threshold;
    int left_count;
};


struct TreeNode
{
    enum class TreeNodeType { NONE, LEAF, NODE };

    TreeNode() :
    type(TreeNodeType::NONE),
    prediction(0),
    threshold(0),
    feature_id(-1),
    left_id(-1),
    right_id(-1) {}

    TreeNodeType type;
    FLOAT prediction;
    FLOAT threshold;
    int feature_id;
    int left_id;
    int right_id;
};


class TreeRegressorMultiMSE
{
public:
    explicit TreeRegressorMultiMSE(int max_depth_, int min_samples_split_, FLOAT min_impurity_decrease_);

    ~TreeRegressorMultiMSE();

    void fit(
        const MatrixFLOAT & X,
        const MatrixFLOAT & G,
        const VectorFLOAT & coefs
    );

    void predict(const MatrixFLOAT & X, VectorFLOAT & pred);

private:

    int fit_node(ArgsType args, int depth);

    std::tuple<ArgsType, ArgsType> split_samples(ArgsType args, Pair pair);

    Pair find_feature_and_threshold(ArgsType args);

    std::tuple<VectorFLOAT, VectorFLOAT> calculate_left_right_loss(ArgsType args);

    FLOAT calculate_loss(ArgsType args, FLOAT pred) const;

    void argsort(ArgsType args, int feature_id) const;

    FLOAT calculate_best_prediction(ArgsType args) const;

    template<typename Derived>
    FLOAT predict_object(Derived & object);

    int max_depth;
    int min_samples_split;
    FLOAT min_impurity_decrease;
    
    std::map<int, TreeNode> tree;
    int current_id;

    const MatrixFLOAT * X_;
    const MatrixFLOAT * G_;
    const VectorFLOAT * coefs_;

    Eigen::VectorX<int> buffer;

    Eigen::VectorX<FLOAT> buffer_left;
    Eigen::VectorX<FLOAT> buffer_right;
    Eigen::VectorX<FLOAT> buffer_num;
    Eigen::VectorX<FLOAT> buffer_1;
    Eigen::VectorX<FLOAT> buffer_2;
};


#endif //TREE_REGRESSOR_HPP
