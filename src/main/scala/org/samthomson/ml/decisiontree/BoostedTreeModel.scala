package org.samthomson.ml.decisiontree

import com.typesafe.scalalogging.LazyLogging
import org.samthomson.ml.Weighted
import org.samthomson.ml.WeightedMean.{Stats => MeanStats}
import org.samthomson.util.StreamFunctions.unfold
import spire.implicits._


case class BoostedTreeModel[K, X, Y](outputSpace: X => Iterable[Y],
                                     xyFeats: FeatureSet.Mixed[K, (X, Y)],
                                     lossFn: TwiceDiffableLoss[Y, Map[Y, Double]],
                                     lambda0: Double,
                                     lambda2: Double,
                                     maxDepth: Int) extends LazyLogging {
  private val regression = RegressionTree(xyFeats, lambda0, maxDepth)

  private def toModel(forest: Vector[Model[(X, Y), Double]]) = MultiClassModel[X, Y](outputSpace, Ensemble(forest))

  def fit(data: Iterable[Example[X, Y]],
          numIterations: Int): (MultiClassModel[X, Y], Double) =
    optimizationPath(data)(Ensemble(Vector[Model[(X, Y), Double]]())).take(numIterations).last

  def optimizationPath(data: Iterable[Example[X, Y]])
                      (initialModel: Model[(X, Y), Double]): Stream[(MultiClassModel[X, Y], Double)] = {
    unfold(Vector(initialModel))({ forest =>
      val (oTree, loss) = fitNextTree(data, toModel(forest))
      oTree.map({ tree =>
        // yield the new tree, and update the "unfold" state to include the new tree
        val newForest = forest :+ tree
        ((toModel(newForest), loss), newForest)
      })
    })
  }

  def fitNextTree(data: Iterable[Example[X, Y]],
                  currentModel: MultiClassModel[X, Y]): (Option[DecisionTree[(X, Y), Double]], Double) = {
    var totalLoss = MeanStats(0.0, 0.0)  // keep track of objective value (loss)
    var n = 0  // number of examples
    // Calculate a 2nd-order Taylor expansion of loss w.r.t. scores, then
    // do the old AdaBoost thing where you turn the quadratic fns into a weighted
    // regression problem.
    logger.debug("calculating residuals")
    val residuals = data.flatMap { case Example(input, goldLabel) =>
      // our forest-so-far produces a score (real number) for each (input, output) pair.
      val scores = currentModel.scores(input).toMap
      // calculate loss and its first two derivatives with respect to scores.
      // hessian is actually a diagonal approximation (i.e. ignores interactions btwn scores).
      val (loss, gradient, hessian) = lossFn.lossGradAndHessian(goldLabel)(scores)
      totalLoss += MeanStats(1.0, loss)
      n += 1
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
    val nextTree = regression.fit(residuals)
    logger.debug(f"loss: ${totalLoss.mean}%10.5f")
    nextTree match {
      case Leaf(avg) if math.abs(avg) <= lambda0 =>
        // don't waste my time with these mickey mouse trees
        logger.debug("Empty tree. Stopping.")
        (None, totalLoss.mean)
      case _ => (Some(nextTree), totalLoss.mean)
    }
  }
}