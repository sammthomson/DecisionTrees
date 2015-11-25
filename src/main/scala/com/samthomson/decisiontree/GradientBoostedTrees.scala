package com.samthomson.decisiontree

import com.samthomson.StreamFunctions.unfold
import com.samthomson.Weighted

import scala.language.implicitConversions


case class RegressionForest[-X](trees: Iterable[DecisionTree[X, Double]]) {
  def predict(input: X): Double = trees.map(_.predict(input)).sum
}

case class MultiClassForest[-X, Y](outputSpace: Iterable[Y], forest: RegressionForest[(X, Y)]) {
  def predict(input: X): Y = scores(input).maxBy(_._2)._1

  def scores(input: X): Iterable[(Y, Double)] = {
    outputSpace.map(o => o -> forest.predict((input, o)))
  }
}


case class GradientBoostedTrees[F, X, Y](outputSpace: Iterable[Y],
                                         xyFeats: FeatureSet.Mixed[F, (X, Y)],
                                         lossFn: TwiceDiffableLoss[Y, Map[Y, Double]],
                                         lambda0: Double,
                                         lambda2: Double,
                                         maxDepth: Int) {

  val regressionModel = RegressionTree(xyFeats, lambda0)

  def fit(data: Iterable[Weighted[Example[X, Y]]],
          numIterations: Int): MultiClassForest[X, Y] = {
    val start = List.empty[DecisionTree[(X, Y), Double]]
    val trees = unfold(start)({ ts => fitNextTree(data, ts).map({ t => (t, t :: ts) }) }).take(numIterations)
    // put trees back in the order they were learned (doesn't really matter)
    MultiClassForest(outputSpace, RegressionForest(trees.toList.reverse))
  }

  def fitNextTree(data: Iterable[Weighted[Example[X, Y]]],
                  currentTrees: List[DecisionTree[(X, Y), Double]]): Option[DecisionTree[(X, Y), Double]] = {
    val forest = MultiClassForest[X, Y](outputSpace, RegressionForest(currentTrees))
    var totalLoss = 0.0  // keep track of objective value (loss)
    // Calculate a 2nd order Taylor expansion of loss w.r.t. scores, then
    // do the old AdaBoost hack where instead of minimizing the quadratic fn directly,
    // you turn it into a weighted regression problem.
    // TODO: just minimize it directly
    val residuals = data.flatMap({ case Weighted(Example(input, goldLabel), w) =>
        // our forest-so-far produces a score (real number) for each (input, output) pair.
        val scores = forest.scores(input).toMap
        // calculate loss and its first two derivatives with respect to scores
        val (loss, gradient, hessian) = lossFn.lossGradAndHessian(goldLabel)(scores)
        totalLoss += loss
        gradient.map({ case (y, grad) =>
          val hess = hessian(y) + lambda2
          Weighted(Example((input, y), -grad / hess), .5 * w * hess) })
      })
    val nextTree = regressionModel.fit(residuals, maxDepth)
    if (nextTree != Leaf(0.0)) {
      println(s"loss: $totalLoss")
      Some(Weighted(nextTree, 1.0))
    } else {
      println("Converged!")
      None
    }
  }
}
