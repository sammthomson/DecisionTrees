package com.samthomson.ml.decisiontree

import com.samthomson.util.StreamFunctions
import StreamFunctions.unfold
import com.samthomson.ml.Weighted
import spire.implicits._


case class BoostedTreeModel[F, X, Y](outputSpace: Iterable[Y],
                                     xyFeats: FeatureSet.Mixed[F, (X, Y)],
                                     lossFn: TwiceDiffableLoss[Y, Map[Y, Double]],
                                     lambda0: Double,
                                     lambda2: Double,
                                     maxDepth: Int) {

  private val regressionModel = RegressionTree(xyFeats, lambda0, maxDepth)

  def fit(data: Iterable[Example[X, Y]],
          numIterations: Int): MultiClassModel[X, Y] = {
    val start = List[Model[(X, Y), Double]]()
    val forest = unfold(start)({ forest =>
      val currentModel = MultiClassModel[X, Y](outputSpace, Ensemble(forest))
      fitNextTree(data, currentModel).map({ tree =>
        // yield the new tree, and update the "unfold" state to include the new tree
        (tree, tree :: forest)
      })
    }).take(numIterations).toVector
    System.err.println(s"Fit ${forest.length} trees.")
    MultiClassModel(outputSpace, Ensemble(forest))
  }

  def fitNextTree(data: Iterable[Example[X, Y]],
                  currentModel: MultiClassModel[X, Y]): Option[DecisionTree[(X, Y), Double]] = {
    var totalLoss = 0.0  // keep track of objective value (loss)
    // Calculate a 2nd-order Taylor expansion of loss w.r.t. scores, then
    // do the old AdaBoost thing where you turn the quadratic fns into a weighted
    // regression problem.
    val residuals = data.flatMap { case Example(input, goldLabel) =>
        // our forest-so-far produces a score (real number) for each (input, output) pair.
        val scores = currentModel.scores(input).toMap
        // calculate loss and its first two derivatives with respect to scores.
        // hessian is actually a diagonal approximation (i.e. ignores interactions btwn scores).
        val (loss, gradient, hessian) = lossFn.lossGradAndHessian(goldLabel)(scores)
        totalLoss += loss
        gradient.map { case (y, grad) =>
          val hess = hessian(y) + lambda2
          // newLoss ~= w * (.5 * hess * theta^2 + grad * theta + oldLoss)    // 2nd-order Taylor approx
          //          = .5 * w * hess * (-grad/hess - theta)^2 + Constant
          //          = weighted squared error of theta w.r.t. -grad / hess
          val argmin = -grad / hess
          val weight = .5 * hess
          Weighted(Example((input, y), argmin), weight)
        }
      }
    val nextTree = regressionModel.fit(residuals)
//    System.err.println(s"loss: $totalLoss")
    if (nextTree != Leaf(0.0)) {
      Some(nextTree)
    } else {
      System.err.println("Empty tree. Stopping.")
      None
    }
  }
}
