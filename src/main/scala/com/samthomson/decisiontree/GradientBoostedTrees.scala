package com.samthomson.decisiontree

import com.samthomson.Weighted

import scala.language.implicitConversions
import scala.math.sqrt


case class RegressionForest[-X](trees: Iterable[Weighted[DecisionTree[X, Double]]]) {
  def predict(input: X): Double = trees.map({ case Weighted(t, w) => w * t.predict(input) }).sum
}

case class MultiClassForest[-X, Y](outputSpace: Iterable[Y], forest: RegressionForest[(X, Y)]) {
  def predict(input: X): Y = scores(input).maxBy(_._2)._1

  def scores(input: X): Iterable[(Y, Double)] = {
    outputSpace.map(o => o -> forest.predict((input, o)))
  }
}


object GradientBoostedTrees {
  def fit[F, X, Y](data: Iterable[Weighted[Example[X, Y]]],
                   outputSpace: Iterable[Y],
                   lossFn: TwiceDiffableLoss[Y, Map[Y, Double]],
                   lambda0: Double,
                   lambda2: Double,
                   maxDepth: Int,
                   numIterations: Int)
                  (implicit xyFeats: FeatureSet.Mixed[F, (X, Y)]): MultiClassForest[X, Y] = {
    val trees =
      (1 to numIterations).foldLeft(List[Weighted[DecisionTree[(X, Y), Double]]]()) { case (ts, i) =>
        val forest = MultiClassForest[X, Y](outputSpace, RegressionForest(ts))
        var totalLoss = 0.0  // keep track of objective value (loss)
        // Calculate a 2nd order Taylor expansion of loss w.r.t. scores, then
        // do the old AdaBoost hack where instead of minimizing the quadratic fn directly,
        // you turn it into a weighted regression problem.
        // TODO: just minimize it directly
        val residuals = data.flatMap({ case Weighted(Example(input, goldLabel), w) =>
          // our forest-so-far produces a score (real number) for each (input, output) pair.
          val scores = forest.scores(input).toMap
//          println(s"scores: $input, $goldLabel, \t $scores")
          // calculate loss and its first two derivatives with respect to scores
          val (loss, gradient, hessian) = lossFn.lossGradAndHessian(goldLabel)(scores)
          totalLoss += loss
  //      println(s"negGrad: \t ${negGradient.map(_.unweighted)}")
//          val lambda2 = 1e-2 * sqrt(i)  // take smaller steps in later iterations
          gradient.map({ case (y, grad) =>
            val hess = hessian(y) + lambda2
            Weighted(Example((input, y), -grad / hess), w * hess) })
        })
        val tree = RegressionTree.fit(residuals, lambda0, maxDepth)
//        println(s"tree: \t $tree")
        if (tree != Leaf(0.0)) {
          println(s"loss: $totalLoss")
          Weighted(tree, 1.0) :: ts
        } else {
          println("Converged!")
          return MultiClassForest(outputSpace, RegressionForest(ts.reverse))
        }
      }
    // put trees back in the order they were learned (doesn't really matter)
    MultiClassForest(outputSpace, RegressionForest(trees.reverse))
  }
}
