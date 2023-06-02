#include <fstream>

#include <cstdlib>
#include <algorithm>
#include <numeric>
#include <exception>

#include "tree_regressor.hpp"



TreeRegressorMultiMSE::TreeRegressorMultiMSE(
    int max_depth_,
    int min_samples_split_,
    FLOAT min_impurity_decrease_
)
: max_depth(max_depth_), min_samples_split(min_samples_split_),
  min_impurity_decrease(min_impurity_decrease_),
  tree(), current_id(0),
  X_(nullptr), G_(nullptr), coefs_(nullptr),
  buffer(), buffer_left(), buffer_right(),
  buffer_num(), buffer_1(), buffer_2() {}


TreeRegressorMultiMSE::~TreeRegressorMultiMSE()
{
}


void TreeRegressorMultiMSE::fit(
        const MatrixFLOAT & X,
        const MatrixFLOAT & G,
        const VectorFLOAT & coefs)
{
    if (!(X.rows() == G.rows() && G.cols() == coefs.size()))
    {
        throw std::runtime_error("Wrong input dimensions");
    }

    tree.clear();
    current_id = 0;

    X_ = &X;
    G_ = &G;
    coefs_ = &coefs;

    Eigen::VectorX<int> args = Eigen::VectorX<int>::LinSpaced(X.rows(), 0, X.rows() - 1);

    buffer.resize(args.size());
    buffer_left.resize(G.rows() * G.cols());
    buffer_right.resize(G.rows() * G.cols());
    buffer_num = Eigen::VectorX<FLOAT>::LinSpaced(X.rows() - 1, 1, X.rows() - 1);
    buffer_1.resize(args.size());
    buffer_2.resize(args.size());

    fit_node(ArgsType(args.data(), args.size()), 0);

    buffer.resize(0);
    buffer_left.resize(0);
    buffer_right.resize(0);
    buffer_num.resize(0);
    buffer_1.resize(0);
    buffer_2.resize(0);
}


int TreeRegressorMultiMSE::fit_node(ArgsType args, int depth)
{
    TreeNode node;
    int node_id = current_id++;

    if ((max_depth != -1 && depth >= max_depth) ||
            (args.size() < min_samples_split))
    {
        node.type = TreeNode::TreeNodeType::LEAF;
        node.prediction = calculate_best_prediction(args);

        tree[node_id] = node;
        return node_id;
    }

    Pair best = find_feature_and_threshold(args);

    if (best.feature_id == -1)
    {
        node.type = TreeNode::TreeNodeType::LEAF;
        node.prediction = calculate_best_prediction(args);

        tree[node_id] = node;
        return node_id;
    }

    auto left_right = split_samples(args, best);
    auto & left = std::get<0>(left_right);
    auto & right = std::get<1>(left_right);

    if (left.size() == 0 || right.size() == 0)
    {
        node.type = TreeNode::TreeNodeType::LEAF;
        node.prediction = calculate_best_prediction(args);

        tree[node_id] = node;
        return node_id;
    }

    node.type = TreeNode::TreeNodeType::NODE;
    node.feature_id = best.feature_id;
    node.threshold = best.threshold;

    node.left_id = fit_node(left, depth + 1);
    node.right_id = fit_node(right, depth + 1);

    tree[node_id] = node;
    return node_id;
}


std::tuple<ArgsType, ArgsType> TreeRegressorMultiMSE::split_samples(ArgsType args, Pair pair)
{
    int size = args.size();
    int right_count = 0;
    
    for (int arg : args)
    {
        if (X_->coeff(arg, pair.feature_id) > pair.threshold)
        {
            right_count++;
        }
    }

    int left_count = size - right_count;
    int left_ind = 0;
    int right_ind = left_count;

    for (int arg : args)
    {
        if (X_->coeff(arg, pair.feature_id) > pair.threshold)
        {
            buffer.coeffRef(right_ind++) = arg;
        }
        else
        {
            buffer.coeffRef(left_ind++) = arg;
        }
    }

    memcpy(args.data(), buffer.data(), sizeof(int) * size);

    return std::make_tuple(
        ArgsType(args.data(), left_count),
        ArgsType(args.data() + left_count, right_count)
    );
}


Pair TreeRegressorMultiMSE::find_feature_and_threshold(ArgsType args)
{
    Pair best;
    best.feature_id = -1;
    FLOAT best_loss;

    FLOAT temp = ((*G_)(args, Eigen::all).colwise().sum() * (*coefs_)).coeff(0);
    FLOAT default_loss = temp * temp / args.size();

    for (int feature_id = 0; feature_id < X_->cols(); feature_id++)
    {
        argsort(args, feature_id);

        auto left_right_loss = calculate_left_right_loss(args);
        auto & left_loss = std::get<0>(left_right_loss);
        auto & right_loss = std::get<1>(left_right_loss);

        for (int thr = 0; thr < args.size() - 1; thr++)
        {
            FLOAT left_val = X_->coeff(args.coeff(thr), feature_id);
            FLOAT right_val = X_->coeff(args.coeff(thr + 1), feature_id);

            if (left_val != right_val)
            {
                FLOAT loss = left_loss.coeff(thr) + right_loss.coeff(thr);

                if (best.feature_id == -1 || loss > best_loss)
                {
                    best_loss = loss;
                    best.feature_id = feature_id;
                    best.threshold = (left_val + right_val) / 2.0;
                    best.left_count = thr;
                }
            }
        }
    }

    if (best.feature_id != -1 && best_loss < default_loss + min_impurity_decrease / coefs_->sum())
    {
        best.feature_id = -1;
        best.threshold = best_loss - default_loss;
    }

    return best;
}


std::tuple<VectorFLOAT, VectorFLOAT> TreeRegressorMultiMSE::calculate_left_right_loss(ArgsType args)
{
    MatrixFLOAT buf_left_1(buffer_left.data(), args.size(), G_->cols());
    MatrixFLOAT buf_right_1(buffer_right.data(), args.size(), G_->cols());
    VectorFLOAT buf_left_2(buffer_1.data(), args.size());
    VectorFLOAT buf_right_2(buffer_2.data(), args.size());
    VectorFLOAT buf_left_3(buffer_1.data(), args.size() - 1);
    VectorFLOAT buf_right_3(buffer_2.data(), args.size() - 1);
    VectorFLOAT buf_num(buffer_num.data(), args.size() - 1);

    auto & coefs = *coefs_;

    buf_left_1 = (*G_)(args, Eigen::all);

    for (auto col : buf_left_1.colwise())
    {
        std::partial_sum(col.begin(), col.end(), col.begin());
    }

    buf_right_1 = buf_left_1.rowwise() - buf_left_1.row(args.size() - 1);

    buf_left_2 = buf_left_1 * coefs;
    buf_right_2 = buf_right_1 * coefs;

    buf_left_2.array() *= buf_left_2.array();
    buf_right_2.array() *= buf_right_2.array();

    buf_left_3.array() /= buf_num.array();

    buf_right_3.reverseInPlace();
    buf_right_3.array() /= buf_num.array();
    buf_right_3.reverseInPlace();

    return std::make_tuple(buf_left_3, buf_right_3);
}



void TreeRegressorMultiMSE::argsort(ArgsType args, int feature_id) const
{
    const auto X = X_;
    
    std::sort(args.data(), args.data() + args.size(), [X, feature_id] (int a, int b)
    {
        return X->coeff(a, feature_id) < X->coeff(b, feature_id);
    });
}


FLOAT TreeRegressorMultiMSE::calculate_best_prediction(ArgsType args) const
{
    auto G = (*G_)(args, Eigen::all);
    auto & coefs = *coefs_;

    return (G.colwise().mean() * coefs).coeff(0) / coefs.sum();
}


FLOAT TreeRegressorMultiMSE::calculate_loss(ArgsType args, FLOAT pred) const
{
    auto G = (*G_)(args, Eigen::all);
    auto & coefs = *coefs_;

    return (
        (G.array() - pred).matrix().colwise().squaredNorm() * coefs
    ).coeff(0);
}


void TreeRegressorMultiMSE::predict(const MatrixFLOAT & X, VectorFLOAT & pred)
{
    for (int i = 0; i < X.rows(); i++)
    {
        pred.coeffRef(i) = predict_object(X.row(i));
    }
}


template<typename Derived>
FLOAT TreeRegressorMultiMSE::predict_object(Derived & object)
{
    int id = 0;
    TreeNode node = tree[id];

    while (node.type != TreeNode::TreeNodeType::LEAF)
    {
        if (object.coeff(node.feature_id) > node.threshold) {
            id = node.right_id;
        } else {
            id = node.left_id;
        }
        node = tree[id];
    }

    return node.prediction;
}

