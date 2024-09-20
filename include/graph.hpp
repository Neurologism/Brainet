#ifndef GRAPH_HPP
#define GRAPH_HPP

#include "variable.hpp"
#include "operation/operation.hpp"

/**
 * @brief The graph class is an implementation of a computational graph. It is used to store the variables and operations and to execute the forward and backward pass.
 */
class Graph 
{
    typedef std::shared_ptr<Variable> VariablePtr;
    typedef std::map<VariablePtr, std::shared_ptr<Tensor<double>>> GradTable;

    std::vector<VariablePtr> mVariableVec; // all variables in the graph
    GradTable mGradTable; // the gradient table for the variables
    
    /**
     * @brief This function builds the gradient table for the variable focus. It is a recursive function that calculates the gradient of the focus variable with respect to all other variables in the graph.
     * To do this, it uses dynamic programming.
     * @param pFocus The variable for which the gradient is calculated.
     * @param gradTable The gradient table that stores already calculated gradients.
     */
    static void mBuildGrad(VariablePtr pFocus, GradTable & gradTable);
    /**
     * @brief This function performs a topological sort on the graph and returns the sorted variables.
     * @return std::vector<VariablePtr> The sorted variables.
     */
    std::vector<VariablePtr> mTopologicalSort( std::vector<VariablePtr> & inputVariables ) const;

public:
    Graph() = default;
    ~Graph() = default;

    /**
     * @brief This function simply executes the operations of the graph in topological order.
     * @param inputVariables The Variables from which the data is propagated through the graph.
     */
    void forward(std::vector<VariablePtr> & inputVariables) const;

    /**
     * @brief This function calculates the gradients of the target variables with respect to the variables in the differentiated vector.
     * It uses the well-known general backpropagation algorithm
     * and is implemented in a dynamic programming fashion.
     * @param targetVariables The variables for which the gradients are calculated.
     * @param outputVariables The target variables are computed with respect to the output variables.
     * @return The gradients of the target variables in the same order as the target variables.
     */
    void backprop(std::vector<VariablePtr> & targetVariables, std::vector<VariablePtr> & outputVariables);

    /**
     * @brief This function returns all variables in the graph.
     * @return std::vector<VariablePtr> The variables in the graph.
     */
    std::vector<VariablePtr> getVariableVec();

    /**
     * @brief This function returns the gradient of a variable.
     * @param pVar The variable for which the gradient is calculated.
     * @return The gradient of the variable.
     */
    std::shared_ptr<Tensor<double>> getGradient(const VariablePtr& pVar);

    /**
     * @brief This function adds a variable to the graph.
     * @param pVar The variable to be added.
     */
    VariablePtr addVariable(const VariablePtr & pVar);

    /**
     * @brief This function removes a variable from the graph.
     * @param pVar The variable to be removed.
     */
    void removeVariable(const VariablePtr & pVar);
};

inline auto GRAPH = std::make_shared<Graph>(); // global graph object
#endif // GRAPH_HPP