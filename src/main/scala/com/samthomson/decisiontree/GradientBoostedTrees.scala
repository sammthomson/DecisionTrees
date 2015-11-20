package com.samthomson.decisiontree

import com.samthomson.Weighted
import spire.implicits._

import scala.language.implicitConversions
import scala.math.sqrt
import com.samthomson.data.TraversableOnceOps._


case class RegressionForest[-X](trees: Iterable[Weighted[DecisionTree[X, Double]]]) {
  def predict(input: X): Double = trees.map({ case Weighted(t, w) => w * t.predict(input) }).sum
}

case class MulticlassForest[-X, Y](outputSpace: Iterable[Y], forest: RegressionForest[(X, Y)]) {
  def predict(input: X): Y = scores(input).maxBy(_._2)._1

  def scores(input: X): Iterable[(Y, Double)] = {
    outputSpace.map(o => o -> forest.predict((input, o)))
  }
}


object GradientBoostedTrees {
  def hingeBoost[F, X, Y]
                (data: Iterable[Weighted[Example[X, Y]]],
                 outputSpace: Iterable[Y],
                 maxDepth: Int,
                 numIterations: Int)
                (implicit xyFeats: FeatureSet.Mixed[F, (X, Y)]): MulticlassForest[X, Y] = {
    val trees =
      (1 to numIterations).foldLeft(List[Weighted[DecisionTree[(X, Y), Double]]]()) { case (ts, i) =>
        val stepSize = 1.0 / sqrt(i)
        val forest = MulticlassForest[X, Y](outputSpace, RegressionForest(ts))
        var totalLoss = 0.0
        val residuals = data.flatMap({ case Weighted(Example(input, goldLabel), w) =>
            // cost-augmented decoding
            val scores = forest.scores(input).toMap
//            println(s"scores: $input, $goldLabel, \t $scores")
            val goldScore = scores(goldLabel) - 1.0
            val augmentedScores = scores.updated(goldLabel, goldScore)
            val predicted = augmentedScores.maxesBy(_._2).map(_._1)
            // gradient is a real number for each (examples, output) pair
            // TODO: update weights of examples
            val negGradient = if (predicted.contains(goldLabel)) {
              outputSpace.map(y => Weighted(Example((input, y), 0.0), w)) // no loss
            } else {
              augmentedScores.map({ case (y, s) =>
                val grad = if (s > goldScore) {
                  totalLoss += math.max(0, s - goldScore)
                  -1.0
                } else if (y == goldLabel) 1.0 else 0.0
                Weighted(Example((input, y), grad * stepSize), w)
              })
            }
//            println(s"negGrad: \t ${negGradient.map(_.unweighted)}")
            negGradient
        })
        val tree = RegressionTree.fit(residuals, maxDepth)
//        println(s"tree: \t $tree")
        if (tree != Leaf(0.0)) {
          println(s"loss: $totalLoss")
          Weighted(tree, 1.0) :: ts
        } else {
          ts
        }
      }
    MulticlassForest(outputSpace, RegressionForest(trees))
  }
}
