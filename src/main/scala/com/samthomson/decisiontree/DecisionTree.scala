package com.samthomson.decisiontree

import com.samthomson.LazyStats.weightedMean
import com.samthomson.{WeightedMse, Weighted}
import spire.algebra.Field
import spire.implicits._

import scala.math.max


case class Example[+X, +Y](input: X, output: Y)


trait Splitter[-X] extends Function[X, Boolean] {
  def choose[T](input: X)(left: T, right: T): T = if (apply(input)) left else right
}
case class FeatureThreshold[+F, V: Ordering](feature: F, threshold: V)(implicit ord: Ordering[V]) extends Splitter[F => V] {
  override def apply(input: F => V): Boolean = ord.lteq(input(feature), threshold)
  override def toString: String = s"$feature <= $threshold"
}


sealed trait DecisionTree[-X, +Y] {
  def depth: Int
  def predict(input: X): Y
}
case class Split[-X, Y](split: Splitter[X],
                        left: DecisionTree[X, Y],
                        right: DecisionTree[X, Y]) extends DecisionTree[X, Y] {
  override val depth = max(left.depth, right.depth) + 1
  final override def predict(input: X): Y = split.choose(input)(left, right).predict(input)
}
case class Leaf[-X, Y](constant: Y) extends DecisionTree[X, Y] {
  override val depth = 1
  final override def predict(input: X): Y = constant
}
object Leaf {
  def averaging[X, Y: Field](examples: Iterable[Weighted[Example[X, Y]]]): Leaf[X, Y] = {
    val weightedOutputs = examples.map(_.map(_.output))
    Leaf(weightedMean(weightedOutputs))
  }
}

object RegressionTree {
  val tolerance = 1e-7

  def fit[F, V: Ordering](data: Iterable[Weighted[Example[F => V, Double]]],
                          features: Iterable[F],
                          maxDepth: Int): DecisionTree[F => V, Double] = {
    val mseStats = WeightedMse.Stats.of(data.map(_.map(_.output)))
    if (maxDepth <= 1 || data.isEmpty || mseStats.meanSquaredError <= tolerance) {
      Leaf.averaging(data)
    } else {
      val (split, error) = bestSplitAndError(data, features)
      val (leftData, rightData) = data.partition(e => split(e.input))
      if (leftData.isEmpty || rightData.isEmpty) {
        Leaf.averaging(data)
      } else {
        val left = fit(leftData, features, maxDepth - 1)
        val right = fit(rightData, features, maxDepth - 1)
        Split(split, left, right)
      }
    }
  }

  // finds the split that minimizes squared error
  private def bestSplitAndError[F, V: Ordering](examples: Iterable[Weighted[Example[F => V, Double]]],
                                                features: Iterable[F]): (Splitter[F => V], Double) = {
    import com.samthomson.WeightedMse.{Stats => MseStats}
    val am = MseStats.hasAdditiveMonoid[Double]

    val splitsAndErrs = features.toSeq.flatMap(feature => {
      val statsByThreshold =
          examples.groupBy(_.input(feature))
              .mapValues(exs => MseStats.of(exs.map(_.map(_.output)))).toVector.sortBy(_._1)
      val splits = statsByThreshold.map(_._1).map(v => FeatureThreshold(feature, v))
      val errors = {
        // errors of left and right side of each split value
        val leftErrors = statsByThreshold.map(_._2).scanLeft(am.zero)(am.plus).tail
        val rightErrors = statsByThreshold.map(_._2).scanRight(am.zero)(am.plus).tail
        for ((l, r) <- leftErrors zip rightErrors) yield {
          l.weight * l.meanSquaredError + r.weight * r.meanSquaredError
        }
      }
      splits zip errors
    })
    splitsAndErrs.minBy(_._2)
  }
}
