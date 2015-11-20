package com.samthomson.decisiontree

import com.samthomson.Weighted
import spire.implicits._

import scala.language.implicitConversions
import scala.math.sqrt
import com.samthomson.data.TraversableOnceOps._


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
                   maxDepth: Int,
                   numIterations: Int)
                  (implicit xyFeats: FeatureSet.Mixed[F, (X, Y)]): MultiClassForest[X, Y] = {
    val trees =
      (1 to numIterations).foldLeft(List[Weighted[DecisionTree[(X, Y), Double]]]()) { case (ts, i) =>
        val stepSize = 0.01 * sqrt(i)
        val forest = MultiClassForest[X, Y](outputSpace, RegressionForest(ts))
        var totalLoss = 0.0

        val residuals = data.flatMap({ case Weighted(Example(input, goldLabel), w) =>
            // cost-augmented decoding
            val scores = forest.scores(input).toMap
//            println(s"scores: $input, $goldLabel, \t $scores")
            // gradient is a real number for each (examples, output) pair
            val (loss, gradient, hessian) = lossFn.lossGradAndHessian(goldLabel)(scores)
            totalLoss += loss
//            println(s"negGrad: \t ${negGradient.map(_.unweighted)}")
            gradient.map({ case (y, grad) =>
//              val hess = hessian(y) + 1e-4
              val hess = hessian(y) + stepSize
              Weighted(Example((input, y), -grad / hess), w * hess) })
        })
        val tree = RegressionTree.fit(residuals, maxDepth)
//        println(s"tree: \t $tree")
        if (tree != Leaf(0.0)) {
          println(s"loss: $totalLoss")
          Weighted(tree, 1.0) :: ts
        } else {
          println("Converged!")
          return MultiClassForest(outputSpace, RegressionForest(ts))
        }
      }
    MultiClassForest(outputSpace, RegressionForest(trees))
  }
}
