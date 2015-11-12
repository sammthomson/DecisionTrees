package com.samthomson.decisiontree

import com.samthomson.LazyStats.weightedMean
import com.samthomson.Weighted
import com.samthomson.decisiontree.GradientBoostedTrees.Conjoined
import spire.algebra.Rig
import spire.implicits._

import scala.language.implicitConversions
import scala.math.sqrt



case class Forest[-X](trees: Iterable[Weighted[DecisionTree[X, Double]]]) {
  def predict(input: X): Double = weightedMean(trees.map(_.map(_.predict(input))))
}

case class BoostedForest[+F, -V: Ordering: Rig, +Y](outputSpace: Iterable[Y], forest: Forest[Either[F, Y] => V]) {
  def predict(input: F => V): Y = scores(input).maxBy(_._2)._1
  def scores(input: F => V): Iterable[(Y, Double)] = {
    outputSpace.map(o => (o, forest.predict(Conjoined(input, o))))
  }
}


object GradientBoostedTrees {
  case class Conjoined[F, V, Y](input: F => V, output: Y)(implicit vRig: Rig[V]) extends (Either[F, Y] => V) {
    override def apply(x: Either[F, Y]): V = x match {
      case Left(f) => input(f)
      case Right(y) => if (y == output) vRig.one else vRig.zero // just need these to be different
    }
  }

  def hingeBoost[F, V: Ordering: Rig, Y]
                (data: Iterable[Weighted[Example[F => V, Y]]],
                 features: Iterable[F],
                 outputSpace: Iterable[Y],
                 maxDepth: Int,
                 numIterations: Int): BoostedForest[F, V, Y] = {
    val featureOrOutput: Iterable[Either[F, Y]] = features.map(Left(_)) ++ outputSpace.map(Right(_))
    var trees = List[Weighted[DecisionTree[Either[F, Y] => V, Double]]]()
    for (i <- 1 to numIterations) yield {
      val stepSize = 1.0 / sqrt(i)
      val f: Forest[(Either[F, Y]) => V] = Forest(trees)
      val forest: BoostedForest[F, V, Y] = BoostedForest(outputSpace, f)
      var loss = 0.0
      val residuals = data.flatMap({
        case Weighted(Example(input, goldLabel), w) =>
          // cost-augmented decoding
          val scores = forest.scores(input).toMap
          val goldScore = scores(goldLabel)
          val augmentedScores = scores.map({ case (y, s) =>
            if (y == goldLabel) (y, s) else (y, s + 1.0)
          })
          val predicted = augmentedScores.maxBy(_._2)._1
          // gradient is a real number for each (examples, output) pair
          val negGradient = if (predicted == goldLabel) {
            outputSpace.map(y => Weighted(Example(Conjoined(input, y), 0.0), w))
          } else {
            augmentedScores.map({ case (y, s) =>
              val grad = if (s > goldScore) -1.0 else if (y == goldLabel) 1.0 else 0.0
              loss += math.max(0, s - goldScore)
              Weighted(Example(Conjoined(input, y), grad * stepSize), w)
            })
          }
          negGradient
      })
//      println(s"loss: $loss")
      val tree = RegressionTree.fit(residuals, featureOrOutput, maxDepth)
      trees = Weighted(tree, 1.0) :: trees
    }
    BoostedForest(outputSpace, Forest(trees))
  }
}
